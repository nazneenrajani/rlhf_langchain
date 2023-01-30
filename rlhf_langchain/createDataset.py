import pandas as pd
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import Repository
from typing import Any

# import llm chain
from langchain import LLMChain

# load env variables
FORCE_PUSH = os.getenv("FORCE_PUSH", "0") == "1"
HF_TOKEN = os.getenv("HF_TOKEN", None)

def loadResponses():
    #check if file exists
    if os.path.isfile('response.csv'):
        df = pd.read_csv('response.csv')
    else:
        df = pd.DataFrame(columns=['response'])
    return df
        
def saveResponsetoDataset(response):
    # load the dataset
    repo = Repository(local_dir="data", clone_from="nazneen/rlhf", use_auth_token=HF_TOKEN)
    if repo.is_repo_clean():
        print("Repo currently clean. Ignoring push_to_hub")
    else:
        # add response to dataset
        df = loadResponses()
        df = df.append({'response': response}, ignore_index=True)
        df.to_csv('response.csv', index=False)

        with repo.commit(commit_message="LLM response", blocking = False):
            repo.git_add("response.csv")
            repo.git_add(auto_lfs_track=True)
            repo.push_to_hub(force=FORCE_PUSH)


# create a wraper that receives a class called LLMChain or LLM and adda functionality when the function predict is called
class RLHFLLMChain(LLMChain):
    # override predict function
    def predict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.
        Args:
            **kwargs: Keys to pass to prompt template.
        Returns:
            Completion from LLM.
        """
        # add new functionality
        # get the prediction
        response = self(kwargs)[self.output_key]
        # save the prediction
        saveResponsetoDataset(response)
        # return the prediction
        return response