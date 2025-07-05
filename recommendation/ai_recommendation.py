from transformers import AutoModelForCausalLM, AutoTokenizer

class AiRecommendation:
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)


    def recommend(self, resume, job_description):
        prompt = f"""
        Given the following resume and job description, provide specific, actionable recommendations to improve the resume so it better matches the job description.

        Resume:
        {resume}

        Job Description:
        {job_description}
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_only = generated_text[len(prompt):].strip()

        return answer_only