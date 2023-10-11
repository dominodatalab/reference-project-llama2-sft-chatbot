# llama2-sft-chatbot

## License
This template is licensed to Customer subject to the terms of the license agreement between Domino and the Customer on file.

## About this project
In this project we demonstrate the use of a pre-trained Large Language Model (LLM) in Domino and the process of fine-tunning the model for a specific task.

![](raw/latest/images/bert.png?inline=true)

Fine-tuning a pre-trained LLM is a commonly used technique for solving NLP problems with machine learning. This is a typical transfer learning task where the final model is realised through a number of training phases:

1. The process typically begins with a pre-trained model, which is not task specific. This model is trained on a large corpora of unlabelled data (e.g. Wikipedia) using masked-language modelling loss. Bidirectional Encoder Representations from Transformers (BERT) is an example for such a model [1].

2. The model undergoes a process of domain specific adaptive fine-tunning, which produces a new model with narrower focus. This new model is better prepared to address domain-specific challenges as it is now closer to the expected distribution of the target data. An example for such a model is FinBERT [2].

3. The final stage of the model building pipeline is the behavioural fine-tuning. This is the step where the model is further fine-tuned using a specific supervised learning dataset. The resulting model is now well equipped to solve one specific problem.

The advantage of using the process outlined above is that pre-training is typically computationally expensive and requires the processing of large volumes of data. Behavioural fine-tuning, on the other hand, is relatively cheap and quick to accomplish, especially if distributed and GPU-accelerated compute can be employed in the process.

In this demo project we use the [Sentiment Analysis for Financial News dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) [3], which provides 4,837 samples of sentiments for financial news headlines from the perspective of a retail investor. This dataset is used in conjuction with FinBERT, which we fine-tune for the purposes of performing sentiment analysis.

The assets available in this project are:

* **llama2_guanaco.ipynb** - A notebook, illustrating the process of getting FinBERT from [Huggingface 🤗](https://huggingface.co/ProsusAI/finbert) into Domino, and using GPU-accelerated backend for the purposes of fine-tuning it with the Sentiment Analysis for Financial News dataset
* **ct2_converter.ipynb** - A script, which performs the same fine-tuning process but can be scheduled and executed as a Domino job. The script also accepts two command line arguments:
    * *lr* - learning rate for the fine-tunning process
    * *epochs* - number of training epochs
* **model.py** - A CSV file containing the Sentiment Analysis for Financial News dataset
* **app_streaming.py** - A scoring function, which is used to deploy the fine-tunned model as a [Domino API](https://docs.dominodatalab.com/en/latest/user_guide/8dbc91/host-models-as-rest-apis/)
* **app.py** - 
* **app.sh** - Launch instructions for the accompanying Streamlit app

## Model API calls

The **model.py** provides a scoring function with the following signature: `generate(prompt)`. To test it, you could use the following JSON payload:

```
{
  "data": {
    "prompt": "What can I do in Paris for a day?"
  }
}
```

# Set up instructions

This project requires the following [compute environments](https://docs.dominodatalab.com/en/latest/user_guide/f51038/environments/) to be present. Please ensure the "Automatically make compatible with Domino" checkbox is selected while creating the environment.

**Environment Base** 

`quay.io/domino/pre-release-environments:project-hub-gpu.main.latest`

**Pluggable Workspace Tools** 
```
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [ "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
 title: "vscode"
 iconUrl: "/assets/images/workspace-logos/vscode.svg"
 start: [ "/opt/domino/workspaces/vscode/start" ]
 httpProxy:
    port: 8888
    requireSubdomain: false
```

Please change the value in `start` according to your Domino version.


You also need to make sure that the hardware tier running the notebook or the fine-tuning script has sufficient resources. A GPU with >=16GB of VRAM is recommended. This project was tested on a V100 with 16GB VRAM


