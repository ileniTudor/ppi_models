import * as tf from "@tensorflow/tfjs";

tf.setBackend("webgl");


const CaseForm = ({ patientId, onCaseAdded }: CaseFormProps) => {

    const [modelName] = useState("yolo11n_detect"); // selected model name
    const [segmentationModelName] = useState("y11n_seg640"); // selected model name
    const [modelLoading, setModelLoading] = useState({ loading: false, progress: 0, message: '' });
    const [loadingInitialSave, setLoadingInitialSave] = useState(false); // Specific for initial save

useEffect(() => {
    let isMounted = true; // Flag to prevent state updates if component unmounts

    const loadAndWarmupModels = async () => {
      if (!isMounted) return;
      setModelLoading({ loading: true, progress: 0, message: "Initializing TensorFlow.js..." });

      try {
        await tf.ready(); // Ensures TensorFlow.js backend is ready
        if (!isMounted) return;
        setModelLoading({ loading: true, progress: 0, message: "Loading models..." });

        const overallStartTime = performance.now();
        let yolo: tf.GraphModel | undefined;
        let yoloSeg: tf.GraphModel | undefined;
        let warmupResults: tf.Tensor | tf.Tensor[] | undefined;
        let warmupResultsSeg: tf.Tensor | tf.Tensor[] | undefined;
        let dummyInput: tf.Tensor | undefined;
        let dummyInputSeg: tf.Tensor | undefined;

        try {
          // --- Load Detection Model (Pathology) ---
          if (!isMounted) return;
          setModelLoading(prev => ({ ...prev, message: `Loading pathology model (${modelName})...`, progress: 0 }));
          let modelLoadStartTime = performance.now();
          yolo = await tf.loadGraphModel(
              `/${modelName}_web_model/model.json`,
              {
                onProgress: (fractions) => {
                  if (isMounted) {
                    setModelLoading(prev => ({ ...prev, progress: fractions, message: `Loading pathology model: ${(fractions * 100).toFixed(0)}%` }));
                  }
                },
              }
          );
          if (!isMounted || !yolo) return;
          let modelLoadEndTime = performance.now();
          console.log(`Pathology model (${modelName}) loaded in ${(modelLoadEndTime - modelLoadStartTime).toFixed(2)} ms`);
          setModelLoading(prev => ({ ...prev, progress: 1, message: `Pathology model loaded. Loading teeth model...` }));

          // --- Load Segmentation Model (Teeth) ---
          if (!isMounted) return;
          setModelLoading(prev => ({ ...prev, message: `Loading teeth model (${segmentationModelName})...`, progress: 0 }));
          modelLoadStartTime = performance.now();
          yoloSeg = await tf.loadGraphModel(
              `/${segmentationModelName}_web_model/model.json`,
              {
                onProgress: (fractions) => {
                  if (isMounted) {
                    setModelLoading(prev => ({ ...prev, progress: fractions, message: `Loading teeth model: ${(fractions * 100).toFixed(0)}%` }));
                  }
                },
              }
          );
          if (!isMounted || !yoloSeg) return;
          modelLoadEndTime = performance.now();
          console.log(`Teeth model (${segmentationModelName}) loaded in ${(modelLoadEndTime - modelLoadStartTime).toFixed(2)} ms`);

          modelLoadStartTime = performance.now();
          // @ts-ignore
          dummyInput = tf.ones(yolo.inputs[0].shape);
          // @ts-ignore
          dummyInputSeg = tf.ones(yoloSeg.inputs[0].shape);
          warmupResults = await yolo.execute(dummyInput) as tf.Tensor | tf.Tensor[];
          warmupResultsSeg = await yoloSeg.execute(dummyInputSeg) as tf.Tensor | tf.Tensor[];

          console.log(`Warmup models in ${(performance.now() - modelLoadStartTime).toFixed(2)} ms`);
          setModelLoading(prev => ({ ...prev, progress: 1, message: "All models loaded. Warming up..." }));

          const totalModelLoadTime = performance.now() - overallStartTime;
          setLoadingTime(totalModelLoadTime.toFixed(2));

          if (!isMounted) return;

          setModelLoading({ loading: false, progress: 1, message: "Models are ready." });

          // @ts-ignore
          setModel({ net: yolo, inputShape: yolo.inputs[0].shape });
          // @ts-ignore
          setSegModel({ net: yoloSeg, inputShape: yoloSeg.inputs[0].shape, outputShape: Array.isArray(warmupResultsSeg) ? warmupResultsSeg.map(e => e.shape) : warmupResultsSeg?.shape
          });

        } finally {
          // Cleanup tensors
          tf.dispose([dummyInput, dummyInputSeg, warmupResults, warmupResultsSeg].filter(t => t  && !(t instanceof Array && t.some(subT => !subT)) )); // More robust dispose
          // If warmupResults or warmupResultsSeg is an array of tensors:
          if (Array.isArray(warmupResults)) warmupResults.forEach(t => t.dispose()); else warmupResults?.dispose();
          if (Array.isArray(warmupResultsSeg)) warmupResultsSeg.forEach(t => t.dispose()); else warmupResultsSeg?.dispose();

          // Do not dispose yolo and yoloSeg here as they are set to state
        }
      } catch (error) {
        if (isMounted) {
          console.error("Error during model loading or warmup:", error);
          let errorMessage = "Failed to load models.";
          if (error instanceof Error) {
            errorMessage = error.message;
          } else if (typeof error === 'string') {
            errorMessage = error;
          }
          setModelLoading({ loading: false, progress: 0, message: `Error: ${errorMessage}` });
        }
      }
    };

    if (modelName && segmentationModelName) { // Only load if model names are set
      loadAndWarmupModels();
    } else {
      setModelLoading({ loading: false, progress: 0, message: "Model names not specified."});
    }

    return () => {
      isMounted = false;
    };
  }, [modelName, segmentationModelName]); // Dependencies

}