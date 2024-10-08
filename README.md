# Bot-Ani? RWKV Twitter Bot Detection Project
Detecting whether users are plants since 2024
<p align="center">
  <img src="assets/twitter_bot_image.png" alt="Twitter Bot Image" width="300"/>
</p>

"Bot-Ani?" is a project aimed at developing a machine learning model for detecting bot accounts on Twitter using the RWKV model, a relatively new architecture that combines the strengths of Recurrent Neural Networks (RNNs) and Transformers. The project involves data collection, preprocessing, feature extraction, model tuning, and evaluation to effectively distinguish between human-operated and bot-operated Twitter profiles.  

## Why is Bot Classification Improtant?
  
Twitter (x.com) is a social media platform in which people can manage a profile, follow and be followed others, and
post texts and images. An estimated 15% of all twitter users are bots - automated accounts. While some are harmless,
others spread fake news on social media, influence the outcome of elections, and propagate conspiracy theories and
harmful ideologies. [2]. We care deeply about the task of bot detection, however due to the size of Twitter and the
elusiveness of several bots, bot moderation must be at least partially automated. 

## Architecture 
We tuned the RWKV architecture and fitted it with a decision head, the hidden-layer embeddings of the model are extracted and concatenated with dimension transformed numerical features (both the ones included in the dataset, and ones we scraped ourselved). These get inputted to a small neural network that provides us with classification probabilities. 
<p align="center">
  <img src="assets/image.png" alt="Twitter Bot Image" width="500"/>
</p>
Our model in code:  

```python
class RWKVForClassification(nn.Module):
    def __init__(self, model, num_labels, num_features):
        super(RWKVForClassification, self).__init__()
        self.model = model

        # Linear layer for numerical features, transforms their dimensions to the size of the hidden state for later concatenation
        self.feature_layer = nn.Linear(num_features, self.model.config.hidden_size)

        # Classification head
        self.classifier = ClassificationHead(self.model.config.hidden_size * 2, num_labels) #a linear layer that takes in the hidden state as well as the concatenated features, and outputs a probability.

    def forward(self, input_ids, attention_mask=None, features=None):
        # Forward pass through the RWKV model for text inputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :].clone()  # Make a copy to avoid in-place operation

        # Forward pass for numerical features
        feature_output = self.feature_layer(features)

        # Concatenate the outputs from the text model and numerical features
        combined_output = torch.cat((cls_output, feature_output), dim=1)

        # Pass through the classification head
        logits = self.classifier(combined_output)
        return logits
```
where the classification head is a simple neural network defined as:  
```python
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
```




## Files


| File                          | Description                                                                                           |
|-------------------------------|-------------------------------------------------------------------------------------------------------|
| **Final Report - Deep Learning Bot-Ani Project.pdf**                 | Final report with an  ethics statement, as well as an architecture description, experimentation, and more. |
| **preprocessing_twitter_bot.ipynb** | Jupyter notebook for preprocessing and feature extraction, as well as importing the data into a Parquet file.    |
| **Preprocessed_dataset.parquet**    | The preprocessed dataset file.                                                                       |
| **RWKV_training.ipynb** | Loading the datafile, RWKV model. Contains the and code for defining the classification model , training it and viewing the results.                          |
| **assets** | Folder containing images and graphs used in this readme.                             |



## Dataset
The dataset we chose to work with is the huggingface AIRT-ML, “Twitter human bots dataset.” [3]. We chose it due to its size (not too big) and that it lets us perform feature extraction, which could prove very helpful. Looking at the objects at our dataset and their extracted textual features, we can see that there certainly are differences between bots and humans that can be used for the classification task.
<p align="center">
  <img src="assets/dist.png" alt="Twitter Bot Image" width="600"/>
</p>

## Training and Results 
The training for classification was done via Cross Entropy loss as a surrogate for 0-1 Loss, since the latter isn't differentiable. The dataset was pre-processed as mentioned prior and then split into a training, validation and test set. For the training we tracked both 0-1 and cross entropy loss for both the training and validation sets, due to long training time we selected and trained the model with different step-sizes for 3 epochs - taking the promising step-size in terms of 0-1 loss in the validation set and training said model further.  
<p align="center">
  <img src="assets/loss_curves_first_3_epochs.png" alt="Twitter Bot Image" width="900"/>
</p>
We trained the model with the chosen learning rate of 2e − 5 for 12 epochs in total - saving the weights every epoch,
and chosen the weights that performed best at epoch 5.
<p align="center">
  <img src="assets/Zero_one_loss_complete.png" alt="Twitter Bot Image" width="600"/>
</p>
The model with the best validation score achieved 0.81 accuracy on the test set, 72% recall and 72% accuracy. To
compare these results seem to be on the same range as the BOTRGCN graph based method that scored an accuracy
of 0.8462 [4], and trained on the Twi-Bot 20 dataset - while 3 percentages short, the Twi-bot dataset contains 200
tweets per labeled object whereas our model uses data found in the profile only. In [5] An LSTM tweet classifier bot
that trained on the PAN2019 dataset achieved 75% accuracy and 75% precision and recall, and BERT tweet classifier
scored an 82 percent accuracy. Interestingly [6] trained a distillBert on the Kaggle bot detection dataset, which
contains pre-processed descriptions and some numerical features, and achieved a precision of 49% on the test set.

By comparing scores we can see that our model performs competitively with other architectures, and sometimes outperforms them in terms of accuracy.
However, It's important to note that there are vast differences between the different datasets, so the comparison isn't direct. Despite that, an 82 percent 
accuracy score is in our opinion very good, and we strongly believe that a larger model variation operating on a bigger dataset can indeed achieve performences on par with- or that exceed those of the current 
dominant architectures. 

## References
[1] Follower-Audit, twitter-gear image https://www.followeraudit.com/blog/how-to-spot-twitter-bots/

[2] E. Ferrara, “Disinformation and social bot operations in the run up to the 2017 french presidential election,” First
Monday, July 2017.

[3] AIRT-ML, “Twitter human bots dataset.” https://huggingface.co/datasets/airt-ml/twitter-human-bots,
2023. Accessed: 2024-08-13.  

[4] S. Feng, H. Wan, N. Wang, and M. Luo, “Botrgcn: Twitter bot detection with relational graph convolutional
networks,” in Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social Networks
Analysis and Mining, ASONAM ’21, ACM, Nov. 2021

[5]  D. Duki´c, D. Keˇca, and D. Stipi´c, “Are you human? detecting bots on twitter using bert,” in 2020 IEEE 7th
International Conference on Data Science and Advanced Analytics (DSAA), pp. 631–636, 2020.

[6] G. Dutta, “Twitter bot detection using distilbert.” https://www.kaggle.com/code/gauravduttakiit/
twitter-bot-detection-distilbert, 2024. Accessed: 2024-08-13.

