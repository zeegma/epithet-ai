<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/zeegma/epithet-ai">
    <img src="https://github.com/user-attachments/assets/58f7e1af-f277-4a2e-a7fe-212bcba418b2" alt="Logo" width="160" height="140">
  </a>

  <h1 align="center">Epithet</h1>
  <p align="center">
    Personality-Driven Username Generator via Genetic-Neuro Ensemble
    <br />
    <a href="https://drive.google.com/file/d/1ewmxv2DuzJIK_xDVT6JWLwcXJXkbdmJz/view?fbclid=IwY2xjawLQr01leHRuA2FlbQIxMQABHo-JZGkMdBVrb0NhoqnBKvyJmvU-mEdnqR8laSbW-iBOC-Zw378GFJQV4ixJ_aem_s9ea9Ckv2Eb72MtVhoOhcw"><strong>Explore the poster »</strong></a>
    <br />
    <br />
    <a href="#demo">View Demo</a>
    ·
    <a href="https://github.com/zeegma/epithet-ai/issues">Report Bug</a>
    ·
    <a href="https://github.com/zeegma/epithet-ai/issues">Request Feature</a>
  </p>
</div>

<!-- PROJECT DEMO -->
## Demo
<div align="center">
  <img src="https://github.com/user-attachments/assets/c78f14e3-09e4-425a-8eec-8001adfef005" alt="Demo" width="80%">
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
This project integrates a neural network and genetic algorithm to generate personalized usernames based on user personality traits. Users start with a personality quiz that quantifies traits into a numerical vector. A neural network then translates these traits into desired username characteristics (e.g., character types, length, tone). A genetic algorithm iteratively evolves username suggestions based on these characteristics, optimizing for highly personalized and creative usernames reflective of user preferences.

<!-- TABLE OF CONTENTS -->
## Table Of Contents
<ol>
  <li>
    <a href="#about-the-project">About The Project</a>
    <ul>
      <li><a href="#table-of-contents">Table Of Contents</a></li>
      <li><a href="#features">Features</a></li>
      <li><a href="#technologies">Technologies Used</a></li>
    </ul>
  </li>
  <li>
    <a href="#application-snapshots">Application Snapshots</a>
  </li>
  <li>
    <a href="#folder-structure">Folder Structure</a>
  </li>
  <li>
    <a href="#contributors">Contributors</a>
  </li>
  <li>
    <a href="#license">License</a>
  </li>
</ol> 

<!-- FEATURES -->
## Features
- **Landing Page**: A visually engaging welcome screen introducing users to Epithet AI, with a clear call-to-action to start the quiz.
- **Personality Quiz Page**: Gathers key personality traits for profiling
- **Genetic Algorithm (GA)**: Optimizes user input to form a refined personality profile
- **Neural Network (NN)**: Predicts the user’s epithet using trained personality data
- **Results Page**: Displays the user’s unique epithet (personality label), includes descriptive traits and is designed for easy sharing or screenshotting

<!-- TECHNOLOGIES USED -->
## Technologies
### Backend & AI
- **[Python](https://www.python.org/)** – Core language for backend logic, neural network, and genetic algorithm implementation.
- **[FastAPI](https://fastapi.tiangolo.com/)** – Web framework for building and handling REST API endpoints.
- **[Keras](https://keras.io/)** / **[TensorFlow](https://www.tensorflow.org/)** – Deep learning libraries used to train and run the personality-to-username neural network model.
- **[PyTorch](https://pytorch.org/)** – Deep learning framework used to train and deploy the neural networks for personality and creativity modeling.

### Frontend
- **[HTML](https://developer.mozilla.org/en-US/docs/Web/HTML)** – Markup language for structuring the UI of the landing page, quiz, and results.
- **[CSS](https://developer.mozilla.org/en-US/docs/Web/CSS)** – Stylesheet language for visual presentation, including layout, fonts, and responsiveness.
- **[JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)** – Scripting language used to handle quiz logic, interactivity, and dynamic UI behavior.

### Development Tools
- **[Git](https://git-scm.com/)** & **[GitHub](https://github.com/)** – Version control and collaborative development.
- **[pipenv](https://pipenv.pypa.io/)** or `requirements.txt` – Dependency management for consistent environments.

<!-- APPLICATION SNAPSHOTS -->
## Application Snapshots
### Landing Page
![Screenshot 2025-07-01 204006](https://github.com/user-attachments/assets/0e02ad0a-0dcf-4010-be99-f5efb5737004)

### Questions
<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
  <img src="https://github.com/user-attachments/assets/192f70ce-024e-4cb5-85f9-34661b74b222" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/71167622-c0b5-4756-b72b-f0264a586089" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/9a6e45b3-79c3-40b2-a5aa-ee7f5794ff0f" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/ad53aaba-3a32-4e4e-8da8-2a131955d577" style="width: 24%; height: auto;"/>
  
  <img src="https://github.com/user-attachments/assets/f53eb248-90ed-422a-9572-c6d3cf3050ff" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/74aaed3e-e4c3-4a41-afda-186e8c024dde" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/c13f380b-4880-4f64-8438-aa631324ffaa" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/b5a7f808-d96a-432e-a128-5a68fe3bb7f8" style="width: 24%; height: auto;"/>

  <img src="https://github.com/user-attachments/assets/9d8257ec-6ab9-445a-9e24-825559e58ac6" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/de628350-338c-4862-9cab-8a225f2cf0a4" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/0ce88a8e-8776-413c-82c9-21a9814ae064" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/038f62cb-2cd3-4ec9-921a-da49bdcfd424" style="width: 24%; height: auto;"/>

  <img src="https://github.com/user-attachments/assets/7d63959e-b05a-4a90-a9db-01f4929c4e2d" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/5b59ab3f-44c1-4e4f-9fbc-d25eca28f3df" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/67fa499e-9b17-408b-a661-01ec547612a3" style="width: 24%; height: auto;"/>
</div>

### User Input
![image](https://github.com/user-attachments/assets/b32da599-ee91-4d9c-8534-0bcc52564040)

### Categories
<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;">
  <img src="https://github.com/user-attachments/assets/9242e861-69b5-41b0-b145-6d87360639a5" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/ca41233f-f69d-4e39-96e7-39fd2dc58ff6" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/8c18b80f-acdf-4adb-8aa0-17d09c088d79" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/86432b64-d319-4311-a3c9-5b6f7af3dc55" style="width: 24%; height: auto;"/>

  <img src="https://github.com/user-attachments/assets/df9a6c4c-9216-4292-8e09-a9a0c5f04cb2" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/b6d82b1e-5872-4ad5-89f0-3b0ee4efb23b" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/65549553-d94a-4247-a346-129efa0300a2" style="width: 24%; height: auto;"/>
  <img src="https://github.com/user-attachments/assets/ed869646-6c96-4eee-9a0b-aed1ce8f2e6c" style="width: 24%; height: auto;"/>
</div>

<!-- FOLDER STRUCTURE -->
## Folder Structure

Here's the folder structure. This is of course not definitive. Feel free to change the files/structures on your own respective folder (again I suggest that you avoid nesting for your own sake)

```
epithet/
│
├── app/                               # Contains main assets and logic for the frontend
|   ├── assets/                        # Static files like icons, images, or fonts
|   ├── data/                          # Preprocessed data, constants, or external files used by the app
|   ├── scripts/                       # JavaScript files controlling UI behavior and interactivity
|   ├── styles/                        # CSS files for styling the interface
├── index.html                         # Entry landing page with the "Start" button
├── name.html                          # Name input for the user.
├── quiz.html                          # Renders the 15-question personality form
├── result.html                        # Shows generated username & explanation
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

<!-- CONTRIBUTOR'S TABLE -->
## Contributors
<table style="width: 100%; text-align: center;">
    <thead>
      <tr>
        <th>Name</th>
        <th>Avatar</th>
        <th>GitHub</th>
        <th>Contributions</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Acelle Krislette Rosales</td>
        <td><img src="https://avatars.githubusercontent.com/u/143507354?v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/krislette">krislette</a></td>
        <td>
          <b>Fullstack Developer:</b> Acelle oversaw the entire development process, created the neural network for personality profiling, and handled the website’s backend.
        </td>
      </tr>
      <tr>
        <td>Regina Bonifacio</td>
        <td><img src="https://avatars.githubusercontent.com/u/116869096?s=400&u=43146b191775802d9ab2f0f721b452ffc52c9efa&v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/feiryrej">feiryrej</a></td>
        <td>
          <b>Frontend Developer:</b> Regina was responsible for designing the landing page and quiz page. She also chose the overall UI theme for both the website and poster.
        </td>
      </tr>
      <tr>
        <td>Henry James Carlos</td>
        <td><img src="https://avatars.githubusercontent.com/u/71052354?v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/hjcarlos">hjcarlos</a></td>
        <td>
          <b>Frontend Developer:</b> Henry was responsible for creating the results page, enhanced the website's overall smoothness, and creating the project poster.
        </td>
      </tr>
      <tr>
        <td>Syruz Ken Domingo</td>
        <td><img src="https://avatars.githubusercontent.com/u/141235021?v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/sykeruzn">sykeruzn</a></td>
        <td>
          <b>Backend Developer:</b> Syke implemented the genetic algorithm (GA) that iteratively evolves candidate usernames and lead the content making process.
        </td>
      </tr>
      <tr>
        <td>Fervicmar Lagman</td>
        <td><img src="https://avatars.githubusercontent.com/u/116869089?v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/perbik">perbik</a></td>
        <td>
          <b>Backend Developer:</b> Fervicmar collaborated on the development of the genetic algorithm, contributing to the design of its core mechanisms, including selection, mutation, and crossover. 
        </td>
      </tr>
      <tr>
        <td>Chrysler Dele Ordas</td>
        <td><img src="https://avatars.githubusercontent.com/u/125347879?v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/soalaluna">soalaluna</a></td>
        <td>
          <b>Technical Writer:</b> Chrysler created the content for the project poster, assisted in its design, and provided the physical board for presentation.
        </td>
      </tr>
      <tr>
        <td>Hans Christian Queja</td>
        <td><img src="https://avatars.githubusercontent.com/u/65350664?v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/HansQueja">HansQueja</a></td>
        <td>
          <b>Backend Developer:</b> Hans contributed to the design of the neural network, focused on its creative logic for mapping personality traits to username characteristics.
        </td>
      </tr>
      <tr>
        <td>Princess Jane Drama</td>
        <td><img src="https://avatars.githubusercontent.com/u/155222986?v=4" alt="" style="border-radius: 50%; width: 50px;"></td>
        <td><a href="https://github.com/pj-drama">pj-drama</a></td>
        <td>
          <b>UI/UX:</b> Princess Jane assisted in designing the website’s user interface and helped with the poster layout. 
        </td>
      </tr>
    </tbody>
  </table>
</section>

<!-- LICENSE -->
## License
Distributed under the [Creative Commons Attribution-NoDerivatives 4.0 International](https://github.com/vitorsr/cc/blob/master/CC-BY-ND-4.0.md) License. See [LICENSE](LICENSE) for more information.

<p align="right">[<a href="#readme-top">Back to top</a>]</p>
