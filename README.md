# Semantic serach of IIT Hyderabad faculties by their research interests
In this project we aim to semantically serach IIT Hyderabad faculties by their stated research interests in their institute webpage

Head over to [Demo app](https://semantic-faculty-recommender.streamlit.app/) and have a look!

The [IIT Hyderabad faculty listing](https://iith.ac.in/people/faculty/#resch) webpage currently allows only exact search by faculty research interest. Below is an example for the keyword "Computer Vision". As you can see only three faculties have Computer Vision as their research interest
![image](https://github.com/ArvindRoshaan/project-semantic-faculty-recommender/assets/91244663/98660005-2822-4bcf-b965-db8718454be9)

We aim to do an upgrade on the exact search using semantic serach and hopefully get this upgrade integrated in the official IIT Hyderabad webpage. The sementic search is performed on the embeddings from a [model](https://huggingface.co/sentence-transformers/allenai-specter) trained on research paper title and abstract. Below is an example from the [demo app](https://semantic-faculty-recommender.streamlit.app/) for the same keyword "Computer Vision"
![image](https://github.com/ArvindRoshaan/project-semantic-faculty-recommender/assets/91244663/f9fd05dc-2e26-4d0c-97b4-5cfa4e44c2a5)
![image](https://github.com/ArvindRoshaan/project-semantic-faculty-recommender/assets/91244663/88253398-203a-4429-a9c7-e0c23c2ee52d)



