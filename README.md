# Sound Classification test
- Predict the label of sounds using pre-trained models.
- Sound input files must be .wav and Models must be .pkl file.

  ### Class
    - There is a total of 8 labels.
    - Dog, Cat, Bird, Lion, Horse, Clap, Scream, Drum
    
  ### Requirements

  #### 1. install python dependencies
  ```
  pip install -r requirements.txt
  ```
  #### 2. Create config file in `/config/config.yaml` 
  Example:
  ```yaml
  local_path : '../sound_files/'
  model_path : 'models/'
  label_ls : ['bird', 'cat', 'clap', 'dog', 'drum', 'horse', 'lion', 'shout']
  ```
  #### 3. Add the files to the assigned folder

  In this case we will put the files inside `sound_files/`
      
  ## How to run:
  ### 1. Predict total songs
      python __init__.py 
  ### 2. Predict a song (Where filename is just the name of the file without the path or extension)
      python __init__.py "filename"
  ### 3. Predict songs using their index (arg1: start index, arg2: end index)
  	  # ex: 1st to 100th song of the list
      python __init__.py 0 100
