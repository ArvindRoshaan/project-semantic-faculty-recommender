# Semantic serach of IIT Hyderabad faculties by their research interests
In this project we aim to semantically serach IIT Hyderabad faculties by their stated research interests in their institute webpage

Head over to [Demo app](https://arvindroshaan-project-sem-researchintsemanticsearchstapp-ecc7qh.streamlit.app/) and have a look!

The [IIT Hyderabad faculty listing](https://iith.ac.in/people/faculty/#resch) webpage currently allows only exact search by faculty research interest. Below is an example for the keyword "Computer Vision". As you can see only three faculties have Computer Vision as their research interest
![image](https://github.com/ArvindRoshaan/project-semantic-faculty-recommender/assets/91244663/98660005-2822-4bcf-b965-db8718454be9)

We aim to do an upgrade on the exact search using semantic serach and hopefully get this upgrade integrated in the official IIT Hyderabad webpage. The sementic search is performed on the embeddings from a [model](https://huggingface.co/sentence-transformers/allenai-specter) trained on research paper title and abstract. Below is an example from the [demo app](https://arvindroshaan-project-sem-researchintsemanticsearchstapp-ecc7qh.streamlit.app/) for the keyword "Deep Learning"
![image](https://github.com/ArvindRoshaan/project-semantic-faculty-recommender/assets/91244663/2c120d16-4bf0-476a-b2fe-f8a131342da2)

