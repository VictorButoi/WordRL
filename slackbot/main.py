import csv
import numpy as np
import os
from os.path import exists

#make a file for the user, if one doesn't already exist, and concat the word + the number of tries it took to solve it 
#python has ways to access a username, use this to make file 

dir_addr = '../data/'

#Check if file exists, if not, create
user_filepath = os.getlogin() + ".json"

if not exists(dir_addr + user_filepath): 
  with open(dir_addr + user_filepath, 'w+'):
    pass

with open(dir_addr + "wordle-answers.txt") as f:
  all_answers = f.read().splitlines() 


with open(dir_addr + "wordle-allowed-guesses.txt") as f:
  all_guesses = f.read().splitlines()


def is_quit():
  ans = input("Quit game?[y/n]: ").casefold()
  while True:
    if ans == 'y':
     return True
    elif ans == 'n':
     return False 
    else:
      ans = input("Invalid response. Quit game?[y/n]: ").casefold()

guess_count = 0 

player_quit = False

answer = all_answers[np.random.randint(low=0, high=len(all_answers))]

while not player_quit: 

  valid_guess = False

  #Parse user guess 
  while not valid_guess:

    usr_guess = input("Guess: ").casefold().replace(" ", "")
    if (usr_guess) in all_guesses: 
      valid_guess = True

  guess_count+= 1

  if usr_guess == answer:
    print("Game won!")
    #print the word with colors in terminal
    #update the JSON file 
    player_quit = is_quit()

  elif guess_count == 6:
    print("Game lost :(")
    #Gameover, failed 
    #write failed into json (word, #guesses, time, failed?)
    #restart option
    player_quit = is_quit()

  else: 
    #print + continue
    pass





