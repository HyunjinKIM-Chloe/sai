# Sound Classification test
- Predict the label of sounds using pre-trained models.
- Sound input files must be .wav and Models must be .pkl file.

  ### Class
    - There is a total of 5 labels.
    - [human, human_voice,	life, nature,	song]
    
  ### Requirements

  #### 1. install python dependencies
  ```
  pip install -r requirements.txt
  ```
  #### 2. Create config file in `/config/config.yaml` 
  Example:
  ```yaml
    video_path : '../sound_files/videos/'
    wav_from_video_path: '../sound_files/wav_from_videos/'
    model_path : 'models/'
    json_path : "jsons/"
    save_path : '../sound_files/wav_cut3s/'
    file_path : '../files/'
  ```
  #### 3. Add the mp4 video files to the assigned folder

  In this case we will put the files inside 'video_path'
