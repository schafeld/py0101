# Python ❤︎ AI
## Doodles in Artificial Intelligence and Machine Learning

Some course work for edX's [IBM PY0101EN Python Basics for Data Science](https://courses.edx.org/courses/course-v1:IBM+PY0101EN+1T2019/course/) and Coursera's [Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow) and code variantions on Tutsplus's ["Learn Machine Learning With Google TensorFlow](https://code.tutsplus.com/courses/learn-machine-learning-with-google-tensorflow/lessons/why-use-tensorflow).

...and/or some Python doodles may be found in this repository. These are my notes as I work on some AI courses, you'll probably not find this stuff very usefull (yet). I do however recommend the original courses for learning about ML and Tensorflow.


#### Notes for *Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning*

Check if you have Tensorflow installed (I use Python 3 so I'm using ```pip3```)

    pip3 show tensorflow

If you don't have it already, install it

    pip3 install tensorflow

[Start](https://jupyter-notebook.readthedocs.io/en/latest/notebook.html#notebook-user-interface) your first Jupyter notebook (courtesy Coursera Introduction to AI course)

    cd intro_T4AI_ML_DL
    jupyter notebook notebook.ipynb

or run the first exercise (house price estimation):

    jupyter notebook Exercise_1_House_Prices_Question.ipynb

In case you have converted the Jupyter notebook into regular Python code (which is easily done with the Visual Studio Code [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) for instance) you can execute it from the terminal as well:

    python3 Exercise_1_House_Prices_Question.py

You can set up a project environment restricted to a current user's terminal workspace with Python's [virtualenv](https://virtualenv.pypa.io/en/stable/userguide/#usage) command:

    pip3 virtualenv
    virtualenv virtualenv_ollis_tensorflow
    cd virtualenv_ollis_tensorflow
    source bin/activate
    pip3 install tensorflow
    pip3 install h5py
    pip3 install pandas

The virtual environment has to be activated as shown above in every new terminal session.
Write installed Python modules into requrements file:

    pip3 freeze > ../reqirements.txt

From that file you could also install modules after you've cloned this repo ([tip source](https://code.tutsplus.com/courses/build-a-web-app-with-the-flask-microframework-for-python/lessons/setting-up-the-environment)):

    pip3 install reqirements.txt




#### Content
Code listed 'in order of appearance', i.e. in the order I played with it.

##### Coursera's ["Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning"](https://www.coursera.org/learn/introduction-tensorflow)
Folder: intro_T4AI_ML_DL/
- notebook.ipynb (HelloWorld doodle)
- Exercise_1_House_Prices_Question.ipynb
- Course_1_Part_4_Lesson_2_Notebook.ipynb
- Course_1_Part_4_Lesson_4_Notebook.ipynb
- Course_1_Part_6_Lesson_2_Notebook.ipynb
- Convolutions_Sidebar.ipynb
- Exercise_3_Question.ipynb (exercise week 3 [teacher solution](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%203%20-%20Convolutions/Exercise%203%20-%20Answer.ipynb))
- Horse_or_Human_NoValidation.ipynb (Note to self: Starting point for project Picturefoods? Doesn't run on my localhost, works on [remote colab](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=RXZT2UsyIVe_) however)
- Course_2_Part_2_Lesson_3_Notebook.ipynb
- Exercise4_Answer.ipynb

##### Tutsplus's ["Learn Machine Learning With Google TensorFlow](https://code.tutsplus.com/courses/learn-machine-learning-with-google-tensorflow/lessons/why-use-tensorflow)

Folder: learn_ML_Tensorflow/
- neural_net_meal_suggestions.py
- neural_net_zoo_animals.py
- neural_net_fruits.py
  Note: I had to ```pip3 install pillow``` because I got an error "ImportError: Could not import PIL.Image. The use of 'array_to_img' requires PIL." (PIL ≙ pillow)

Data sources used (Kaggle grants [Database: Open Database, Contents: Database Contents](https://opendatacommons.org/licenses/dbcl/1.0/) license, see their website for details):

- [Zoo animal classification](https://www.kaggle.com/uciml/zoo-animal-classification)
- [Fruits 360 dataset](https://www.kaggle.com/moltean/fruits/downloads/fruits.zip/49) (not included in repo, download from Kaggle)


#### Acknowledgement
- Notebooks for "[Coursera's Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](https://www.coursera.org/learn/introduction-tensorflow)" can be found on [Github](https://github.com/lmoroney/dlaicourse).

☞ If you're a fellow Coursera or edX student please keep in mind that you pledged to respect the honor code. Don't copy other people's course work – your future self will be grateful to you.

to be continued...