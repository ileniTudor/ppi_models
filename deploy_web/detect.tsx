import * as tf from "@tensorflow/tfjs";
// Assuming detect_labels.json is correctly imported as a default export
import labels from "./detect_labels.json";

const numClass: number = labels.length;

// Define types for the preprocess function parameters and return value
type PreprocessResult = [tf.Tensor, number, number]; // [inputTensor, xRatio, yRatio]

const preprocess = (source: HTMLImageElement, modelWidth: number, modelHeight: number): PreprocessResult => {
    const img = tf.browser.fromPixels(source); // Create img tensor
    const [h, w] = img.shape.slice(0, 2); // Get source width and height
    const maxSize = Math.max(w, h); // Get max size

    // xRatio and yRatio are now calculated immediately and are guaranteed to be assigned
    const xRatio: number = maxSize / w;
    const yRatio: number = maxSize / h;

    const input = tf.tidy(() => {
        // padding image to square => [n, m] to [n, n], n > m
        const imgPadded: tf.Tensor3D = img.pad([
            [0, maxSize - h], // padding y [bottom only]
            [0, maxSize - w], // padding x [right only]
            [0, 0], // padding for channels
        ]) as tf.Tensor3D; // Cast to Tensor3D

        return tf.image
            .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
            .div(255.0) // normalize
            .expandDims(0); // add batch
    });

    tf.dispose(img); // Dispose the intermediate 'img' tensor

    return [input, xRatio, yRatio];
};

const MODEL_INPUT_WIDTH: number = 640;
// MODEL_INPUT_HEIGHT is not directly used in the logic, so it can be removed or kept as a comment.
// const MODEL_INPUT_HEIGHT: number = 640;

// Define the return type for the detect function
interface DetectionResult {
    boxes: number[][];
    scores: number[];
    classes: number[];
}

export const detect = async (
    source: HTMLImageElement,
    model: any, // You might want to define a more specific type for your TF.js model if available
    callback: () => void = () => {}
): Promise<DetectionResult> => {
    // Ensure TensorFlow.js and its backend are ready
    await tf.ready();

    const [modelWidth, modelHeight]: number[] = model.inputShape.slice(1, 3); // get model width and height

    tf.engine().startScope(); // start scoping tf engine
    // Destructure only 'input' as xRatio_preprocess and yRatio_preprocess are not used in detect
    const [input]: [tf.Tensor, number, number] = preprocess(source, modelWidth, modelHeight); // preprocess image


    // Changed: Assuming model.net.execute(input) returns a single tf.Tensor directly, not an array.
    const res: tf.Tensor = model.net.execute(input); // inference model
    const transRes: tf.Tensor = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]

    // Use tf.tidy to manage tensor disposal within this block for synchronous operations
    const [boxes, scores, classes]: [tf.Tensor2D, tf.Tensor1D, tf.Tensor1D] = tf.tidy(() => { // nmsResult is now handled outside tidy
        const w: tf.Tensor = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
        const h: tf.Tensor = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
        const x1: tf.Tensor = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
        const y1: tf.Tensor = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
        const boxesTensor: tf.Tensor2D = tf // Explicitly type as Tensor2D
            .concat(
                [
                    y1,
                    x1,
                    tf.add(y1, h), //y2
                    tf.add(x1, w), //x2
                ],
                2
            )
            .squeeze<tf.Tensor2D>(); // process boxes [y1, x1, y2, x2] and squeeze to Tensor2D

        // class scores
        // rawScores will be [1, num_detections, numClass] before squeeze(0) -> [num_detections, numClass] (Tensor2D)
        const rawScores: tf.Tensor2D = transRes.slice([0, 0, 4], [-1, -1, numClass]).squeeze<tf.Tensor2D>(); // Squeeze all dims of size 1, then assert as Tensor2D
        const [scoresTensor, classesTensor]: [tf.Tensor1D, tf.Tensor1D] = [rawScores.max(1), rawScores.argMax(1)]; // get max scores and classes index

        // Dispose intermediate tensors created within this tidy block explicitly if needed,
        // though tf.tidy should handle most that are not returned.
        tf.dispose([rawScores, w, h, x1, y1]);

        // Return tensors needed outside tidy for NMS and data retrieval
        return [boxesTensor, scoresTensor, classesTensor];
    }); // end tf.tidy

    // Perform NMS asynchronously outside tf.tidy to avoid UI lock
    const nmsResult: tf.Tensor1D = await tf.image.nonMaxSuppressionAsync(
        boxes, // Now correctly typed as Tensor2D
        scores,
        1000, // maxBoxes (example value, adjust as needed)
        0.5, // iouThreshold (example value, adjust as needed)
        0.25 // scoreThreshold (example value, adjust as needed)
    );

    // --- Retrieve data asynchronously after NMS ---
    let boxes_data: number[][], scores_data: number[], classes_data: number[];
    try {
        // Gather data using nmsResult indices
        const nmsIndices: number[] = await nmsResult.array(); // Get the indices from the nmsResult tensor

        // Gather the boxes, scores, and classes based on the NMS indices
        const finalBoxes: tf.Tensor2D = boxes.gather(nmsIndices, 0); // Explicitly type as Tensor2D
        const finalScores: tf.Tensor1D = scores.gather(nmsIndices, 0); // Explicitly type as Tensor1D
        const finalClasses: tf.Tensor1D = classes.gather(nmsIndices, 0); // Explicitly type as Tensor1D

        // Get the data as arrays with explicit type casting
        boxes_data = await finalBoxes.array() as number[][];
        scores_data = await finalScores.array() as number[];
        classes_data = await finalClasses.array() as number[];

        // Dispose of the tensors created in this block
        tf.dispose([nmsResult, finalBoxes, finalScores, finalClasses]);

    } catch (error: any) { // Catch error with 'any' type for flexibility
        console.error("Error retrieving data from tensors after NMS:", error);
        // Dispose tensors if data retrieval fails
        tf.dispose([boxes, scores, classes, res, transRes, input]); // Ensure all tensors are disposed
        tf.engine().endScope();
        callback();
        return { boxes: [], scores: [], classes: [] }; // Return empty arrays on error
    } finally {
        // Dispose of tensors returned from tidy that were not disposed in the try block
        // 'boxes', 'scores', 'classes' are now disposed here after their data is gathered
        tf.dispose([boxes, scores, classes]);
    }

    // --- Scale the boxes to the original image size ---
    const originalImageWidth: number = source.width;
    const originalImageHeight: number = source.height;

    // Calculate the maximum dimension of the original image (used for padding in preprocess)
    const maxSize: number = Math.max(originalImageWidth, originalImageHeight);

    // Calculate the scaling factor from the model input size to the padded square size
    // Since padding is applied to make the smaller dimension equal to the larger dimension,
    // the scaling factor is the ratio of the padded square size (maxSize) to the model input size.
    const scaleFactor: number = maxSize / MODEL_INPUT_WIDTH; // Assuming MODEL_INPUT_WIDTH === MODEL_INPUT_HEIGHT

    // Scale the box coordinates from model input space (640x640) to the original image size (1615x840)
    // Since padding is only on the bottom and right, the top-left corner aligns.
    const scaled_boxes_data: number[][] = boxes_data.map(box => {
        const [y1_model, x1_model, y2_model, x2_model] = box;
        const x1_original: number = x1_model * scaleFactor;
        const y1_original: number = y1_model * scaleFactor;
        const x2_original: number = x2_model * scaleFactor;
        const y2_original: number = y2_model * scaleFactor;
        return [y1_original, x1_original, y2_original, x2_original];
    });

    // Dispose of tensors created outside or explicitly returned from tidy that are no longer needed
    tf.dispose([res, transRes, input]); // 'res', 'transRes', 'input' are the remaining top-level tensors

    callback();

    tf.engine().endScope(); // end of scoping

    // Return the scaled boxes, scores, and classes
    return { boxes: scaled_boxes_data, scores: scores_data, classes: classes_data };
};
