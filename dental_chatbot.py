"""
Dental Pathology Training Chatbot
A conversational system for dental students to practice pathology diagnosis
"""

from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from token_file import token


class DentalPathologyBot:
    def __init__(self, model_name):
        """
        Initialize the chatbot with a conversational LLM model
        
        Args:
            model_name: HuggingFace model identifier
            
        Recommended models for 6GB VRAM:
            - "microsoft/Phi-3-mini-4k-instruct" (3.8B params, ~7.5GB download, excellent quality)
            - "google/gemma-2-2b-it" (2B params, ~5GB download, good quality)
            - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params, ~2.2GB download, fast)
        """
        print(f"Loading model: {model_name}...")
        print("This may take a few minutes on first run...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=token
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {self.device}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
        
        self.conversation_history = []
        self.pathology_data = {}
        
    def set_pathology_case(self, pathology: str, manifestations: Dict):
        """
        Set up a specific pathology case for the student to diagnose
        
        Args:
            pathology: The actual diagnosis (hidden from student)
            manifestations: Dictionary of symptoms and patient information
        """
        self.pathology_data = {
            "diagnosis": pathology,
            "manifestations": manifestations
        }
        self.conversation_history = []
        
        # Create the system prompt
        self.system_prompt = self._create_system_prompt(pathology, manifestations)
        
    def _create_system_prompt(self, pathology: str, manifestations: Dict) -> str:
        """Create the system prompt that instructs the model"""
        
        prompt = f"""You are roleplaying as a dental patient. You have {pathology}, but you must NEVER reveal this diagnosis name to the dental student.

PATIENT INFORMATION:
- Chief Complaint: {manifestations.get('chief_complaint', 'tooth pain')}
- Pain Description: {manifestations.get('pain', 'moderate discomfort')}
- Location: {manifestations.get('location', 'not specified')}
- Duration: {manifestations.get('duration', 'recent onset')}
- Aggravating Factors: {manifestations.get('aggravating_factors', 'none noted')}
- Relieving Factors: {manifestations.get('relieving_factors', 'none noted')}
- Visual Appearance: {manifestations.get('appearance', 'normal')}
- Medical History: {manifestations.get('medical_history', 'unremarkable')}
- Additional Info: {manifestations.get('additional_info', 'none')}

INSTRUCTIONS:
1. Answer the dental student's questions as this patient would
2. Describe sensations, pain, what you see/feel in your mouth
3. Be consistent with the pathology symptoms above
4. Provide helpful details when asked about:
   - When symptoms started and how they progressed
   - What makes it better or worse
   - What it looks like or feels like
   - Previous dental work or medical conditions
5. NEVER say the name "{pathology}" or give away the diagnosis directly
6. If the student seems stuck, you can provide gentle hints about unusual symptoms
7. Be conversational and natural, like a real patient
8. Keep responses concise (2-4 sentences typically)

Respond naturally as this patient would."""

        return prompt
    
    def chat(self, student_message: str) -> str:
        """
        Process a message from the student and return patient response
        
        Args:
            student_message: The question or statement from the dental student
            
        Returns:
            The patient's (model's) response
        """
        # Build the conversation with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["student"]})
            messages.append({"role": "assistant", "content": turn["patient"]})
        
        # Add current message
        messages.append({"role": "user", "content": student_message})
        
        # Format for the model
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback for models without chat template
            formatted_prompt = f"{self.system_prompt}\n\n"
            for turn in self.conversation_history:
                formatted_prompt += f"Student: {turn['student']}\nPatient: {turn['patient']}\n\n"
            formatted_prompt += f"Student: {student_message}\nPatient:"
        
        # Generate response
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.pad_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Clean up response (remove any continuation prompts)
        if "Student:" in response:
            response = response.split("Student:")[0].strip()
        
        # Store in conversation history
        self.conversation_history.append({
            "student": student_message,
            "patient": response
        })
        
        return response
    
    def reveal_diagnosis(self) -> Dict:
        """
        Reveal the actual diagnosis and case details
        
        Returns:
            Dictionary with diagnosis and manifestations
        """
        return {
            "diagnosis": self.pathology_data["diagnosis"],
            "manifestations": self.pathology_data["manifestations"],
            "conversation_turns": len(self.conversation_history)
        }
    
    def reset(self):
        """Reset the conversation"""
        self.conversation_history = []


# Example pathology cases
EXAMPLE_CASES = {
    "periapical_abscess": {
        "pathology": "Periapical Abscess",
        "manifestations": {
            "chief_complaint": "severe tooth pain",
            "pain": "intense, throbbing, constant pain that worsens with pressure",
            "location": "lower right first molar",
            "duration": "started 3 days ago, getting progressively worse",
            "aggravating_factors": "chewing, hot foods, tapping on the tooth",
            "relieving_factors": "cold water provides temporary relief",
            "appearance": "slight swelling of the gum near the tooth, redness",
            "medical_history": "no significant medical history, had a filling on this tooth 6 months ago",
            "additional_info": "patient reports difficulty sleeping due to pain, takes ibuprofen with minimal relief"
        }
    },
    "oral_candidiasis": {
        "pathology": "Oral Candidiasis (Thrush)",
        "manifestations": {
            "chief_complaint": "white patches in mouth",
            "pain": "mild burning sensation, uncomfortable",
            "location": "tongue and inner cheeks",
            "duration": "noticed about a week ago",
            "aggravating_factors": "spicy or acidic foods",
            "relieving_factors": "rinsing with water helps temporarily",
            "appearance": "white, cottage cheese-like patches that can be wiped off leaving red areas",
            "medical_history": "recently completed antibiotic treatment for bronchitis, type 2 diabetes",
            "additional_info": "altered taste sensation, dry mouth"
        }
    },
    "aphthous_ulcer": {
        "pathology": "Aphthous Ulcer (Canker Sore)",
        "manifestations": {
            "chief_complaint": "painful sore in mouth",
            "pain": "sharp, stinging pain, especially when eating or talking",
            "location": "inside lower lip",
            "duration": "appeared 2 days ago",
            "aggravating_factors": "acidic foods, touching it with tongue",
            "relieving_factors": "topical numbing gel, avoiding irritants",
            "appearance": "small round ulcer with yellow-white center and red border",
            "medical_history": "gets these occasionally, especially when stressed",
            "additional_info": "no fever, not spreading, similar to previous sores"
        }
    }
}


def main():
    """Main function to run the dental pathology training chatbot"""


    print("=" * 60)
    print("DENTAL PATHOLOGY TRAINING CHATBOT")
    print("=" * 60)
    print("\nInitializing chatbot...")
    
    # Initialize the bot (you can change the model here)
    # Other good options: "meta-llama/Llama-2-7b-chat-hf", "HuggingFaceH4/zephyr-7b-beta"
    # bot = DentalPathologyBot(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    bot = DentalPathologyBot(model_name="google/gemma-2-2b-it")

    # Select a case
    print("\n" + "=" * 60)
    print("AVAILABLE CASES:")
    for i, (key, case) in enumerate(EXAMPLE_CASES.items(), 1):
        print(f"{i}. {case['pathology']}")
    
    while True:
        try:
            choice = input("\nSelect a case number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return
            
            case_key = list(EXAMPLE_CASES.keys())[int(choice) - 1]
            selected_case = EXAMPLE_CASES[case_key]
            break
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
    
    # Set up the pathology case
    bot.set_pathology_case(
        pathology=selected_case["pathology"],
        manifestations=selected_case["manifestations"]
    )
    
    print("\n" + "=" * 60)
    print("SCENARIO START")
    print("=" * 60)
    print("\nA patient enters your clinic. Begin your examination by asking questions.")
    print("Commands: 'reset' - start over, 'reveal' - show diagnosis, 'quit' - exit\n")
    
    # Conversation loop
    while True:
        try:
            student_input = input("\n[YOU]: ").strip()
            
            if not student_input:
                continue
            
            if student_input.lower() == 'quit':
                print("\nExiting chatbot. Good luck with your studies!")
                break
            
            if student_input.lower() == 'reset':
                bot.reset()
                print("\n[SYSTEM] Conversation reset. Start fresh with your questions.")
                continue
            
            if student_input.lower() == 'reveal':
                result = bot.reveal_diagnosis()
                print("\n" + "=" * 60)
                print("DIAGNOSIS REVEALED")
                print("=" * 60)
                print(f"\nActual Diagnosis: {result['diagnosis']}")
                print(f"Conversation Turns: {result['conversation_turns']}")
                print("\nManifestation Details:")
                for key, value in result['manifestations'].items():
                    print(f"  - {key.replace('_', ' ').title()}: {value}")
                print("\n" + "=" * 60)
                
                cont = input("\nStart a new case? (y/n): ").strip().lower()
                if cont == 'y':
                    return main()
                else:
                    break
            
            # Get patient response
            print("\n[PATIENT]: ", end="", flush=True)
            response = bot.chat(student_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nExiting chatbot. Good luck with your studies!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Try rephrasing your question.")


if __name__ == "__main__":
    main()
