<h1 align="center">Ipitit</h1>

## Intro

Please put your files on their respective folders to avoid conflict and/or confusion. As much as possible, don't nest folders - this is for your sanity I sweah

## Folder Structure

Here's the folder structure. This is of course not definitive. Feel free to change the files/structures on your own respective folder (again I suggest that you avoid nesting for your own sake)

```
epithet/
│
├── app/                               # Streamlit app
│   ├── ui/
│   │   ├── quiz_form.py               # Renders the 15-question personality form
│   │   ├── result_display.py          # Shows generated username & explanation
│   │   ├── utils.py                   # Streamlit-specific helpers (session, etc)
│   ├── main.py                        # Entry point: streamlit run app/main.py
│
├── core/                              # Core logic: models + GA
│   ├── creativity_nn.py               # Load/use trained NN2
│   ├── gen_algo.py                    # Genetic algo
│   ├── personality_nn.py              # Load/use trained NN1
│   ├── preprocess.py                  # Vector/text preprocessing
│   ├── word_pools.py                  # Word pools ofc
│
├── models/                            # Trained model weights
│   ├── personality_model.pt           # Future contents (txt for now)
│   ├── creativity_model.pt            # Future contents (txt for now)
│
├── training/                          # Training code (not part of Streamlit app)
│   ├── gen_algo/                      # Base training/logic for GA
│   │   ├── main.py
│   │   ├── sample_dataset.txt
│   │   ├── train.py
│   ├── nn_creativity/                 # Base training for creativity NN
│   │   ├── main.py
│   │   ├── creativity_dataset.txt
│   │   ├── train.py
│   ├── nn_personality/                # Base training for personality NN
│   │   ├── main.py
│   │   ├── train.py
│
├── .gitignore
├── README.md                          # This README
├── requirements.txt                   # torch, streamlit, numpy, etc.
```
