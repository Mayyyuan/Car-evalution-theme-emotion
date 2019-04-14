# Car evalution-Theme&emotion
CCF 2018: Analyze themes and emotions through user's evaluation of the car.

  data:
      Theme: 10 categories:
          price, configuration, interior, power, safety, fuel consumption, handling, space, comfort, appearance.
      Emotion: 3 categories:
          -1,0,1.
      
  model:
    CNN;
    
  included：
    Pre-trained word2vector, 
    shuffle————increase the amount of data,
    dictionary generalization, including car brand, location, time, money, etc.
    Multi-target recognition;
   
  main file: You can train and predict by running this file.
  util file: This file customizes a number of called functions,
             including data processing functions, vocabulary generalization functions, etc.
  model file: This file is used to configure the network architecture.
