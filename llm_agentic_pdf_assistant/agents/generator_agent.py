from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

class GeneratorAgent:
    """ 
    A class to generate responses using the TinyLlama model.
    """
    
    def __init__(self, model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_path = "./local_model/", save_model = True):
        """
        Initialize the generator agent with a model name and local path.
        
        Args:
            model_name (str): The name of the model to load.
            local_path (str): The local path to save/load the model.
        """
        self.model_name = model_name
        self.local_model_path = local_path + model_name.split("/")[-1]
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Load or download the model
        if os.path.exists(self.local_model_path) and \
           os.path.exists(os.path.join(self.local_model_path, "pytorch_model.bin")) and \
           os.path.exists(os.path.join(self.local_model_path, "tokenizer_config.json")):
            self.tokenizer = self._load_tokenizer(self.local_model_path)
            self.model = self._load_model(self.local_model_path)
        else:
            self.tokenizer = self._download_tokenizer(self.model_name)
            self.model = self._download_model(self.model_name)

            if save_model:
                self._save_model_locally(self.local_model_path)

        self.device_index = 0 if self.device == "cuda" else 1
        # Load the pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_index
        )
    def _load_tokenizer(self, local_path):
        """
        Load the tokenizer from the local path.
        """
        try:
            print("✅ Loading tokenizer from local folder...")
            return AutoTokenizer.from_pretrained(local_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return None
    def _load_model(self, local_path):
        """
        Load the model from the local path.
        """
        try:
            print("✅ Loading model from local folder...")
            return AutoModelForCausalLM.from_pretrained(local_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    def _download_tokenizer(self, model_name):
        """
        Download the tokenizer from Hugging Face.
        """
        try:
            print("⬇️  Downloading tokenizer from Hugging Face...")
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error downloading tokenizer: {e}")
            return None
        
    def _download_model(self, model_name):
        """
        Download the model from Hugging
        """
        try:
            print("⬇️  Downloading model from Hugging Face...")
            return AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Error downloading model: {e}")
    
    def _save_model_locally(self, local_path):
        # Save the model locally for future use
        try:
            print("💾 Saving model locally...")
            self.tokenizer.save_pretrained(local_path)
            self.model.save_pretrained(local_path)
        except Exception as e:
            print(f"Error saving model locally: {e}")
        
    def generate_response(self, context, question):
        """
        Generate a response to a question based on the context using the TinyLlama model.
        
        Args:
            context (str): The context in which to generate the response.
            question (str): The question to answer.
        
        Returns:
            str: The generated response.
        """
        # Prepare the input for the model
        prompt = f"""[INST] <<SYS>> You are a helpful assistant, in context I pass embeddings of
                    a Research Paper or a document, read it and answer questions related to the 
                    paper. <</SYS>>

                \nContext:
                {context}\n

                Question: {question}\n
                Answer:
                [/INST]
                """
        
        # Generate the response
        response = self.generator(prompt, 
                            max_new_tokens=300, 
                            truncation = True, 
                            do_sample = True, 
                            temperature = 0.7, 
                            top_p = 0.95, 
                            num_return_sequences=1)
        
        # Extract and return the generated text
        return response[0]['generated_text'].split("Answer:")[-1].strip()

