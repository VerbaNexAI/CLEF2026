#%%
import dspy
from dotenv import load_dotenv
load_dotenv()
import os
from typing import Literal
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces=True,
)

lm = dspy.LM(
    model="openrouter/openai/gpt-oss-120b",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API"),
    temperature=1,
)
dspy.configure(lm=lm)

class AtClassification(dspy.Signature):
    """
    Analyze the text and determine whether there is evidence that the person was at the specified location at any time before the document’s publication date. 
    Use NLP techniques if necessary to interpret the text and identify relevant contextual or temporal clues indicating presence.
    """

    text = dspy.InputField(desc="text")
    person = dspy.InputField(desc="person")
    location = dspy.InputField(desc="The location")
    date = dspy.InputField(desc="The document publication date")

    classification: Literal["TRUE", "FALSE", "PROBABLE"] = dspy.OutputField()
    explanation = dspy.OutputField(desc="Brief explanation for the classification")



class PresenceDetection(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(AtClassification)

    def forward(self, text, person, location, date):
        return self.predict(
            text=text,
            person=person,
            location=location,
            date=date
        )


program = PresenceDetection()

# load optimized state
program.load("./gepa_at_gpt-oss-20b.json")

program.predict.lm = lm


result = program(
    text="Napoleon arrived in Paris in March 1815",
    person="Napoleon",
    location="Bogotá",
    date="1815-03-10",
)

print(result.classification)
# %%
import dspy

lm = dspy.LM("openai/unsloth/lfm2.5-1.2b-instruct",
                 api_base="http://127.0.0.1:1234/v1",  # ensure this points to your port
                 api_key="")
dspy.configure(lm=lm)
# %%
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']