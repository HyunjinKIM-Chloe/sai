# Sounds Classification test
- Predict the label of sounds with pre-trained models.
- Sounds input type should be .wav and Model should be .pkl file.

  ### Class
    - Total 8 labels
    - Dog, Cat, Bird, Lion, Horse, Clap, Scream, Drum
    
  ### install requirements
       pip install -r requirements.txt
    
  ## How to run scripts
  ### 1. Predict total songs
      python __init__.py 
  ### 2. Predict a song (arg1: Wav filename - should be string type)
      python __init__.py "filename"
  ### 3. Predict songs with indexing (arg1: start index, arg2: end index)
  	  # ex: 1st to 100th song of the list
      python __init__.py 0 100
