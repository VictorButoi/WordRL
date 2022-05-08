import csv
import numpy as np
import os
from os.path import exists

#make a file for the user, if one doesn't already exist, and concat the word + the number of tries it took to solve it 
#python has ways to access a username, use this to make file 


class GameColor:
  MISPLACED = "\033[1;33m"
  CORRECT = "\033[0;32m"
  INCORRECT = "\033[1;30m"
  END = "\033[0m"

#Replace all the interactions from the terminal into slack

dir_addr = '../data/'

#Check if file exists, if not, create
user_filepath = "guesses.json"

if not exists(dir_addr + user_filepath): 
  with open(dir_addr + user_filepath, 'w+'):
    pass

#CHANGE: Answers and guesses use the same file: 5_words.txt

with open(dir_addr + "5_words.txt") as f:
  all_answers = f.read().splitlines() 


with open(dir_addr + "5_words.txt") as f:
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


def sample_game():
  all_answers[np.random.randint(low=0, high=len(all_answers))]


def print_colors(ans, guess): 
  #MAKE 
  pass
  

guess_count = 0 

player_quit = False

answer = sample_game()

while not player_quit: 

  valid_guess = False

  #Parse user guess 
  while not valid_guess:

    usr_guess = input("Guess: ").casefold().replace(" ", "")
    if (usr_guess) in all_guesses: 
      valid_guess = True
    else:
      print("Retry, invalid guess")

  guess_count+= 1

  if usr_guess == answer:
    print("Game won!")
    print_colors(answer, usr_guess)
    #print the word with colors in terminal
    #update the JSON file 
    player_quit = is_quit()
    answer = sample_game()

  elif guess_count == 6:
    print_colors(answer, usr_guess)
    print("Game Lost :(")
    print("Answer: " + answer)
    #Gameover, failed 
    #write failed into json (word, #guesses, time, failed?)
    player_quit = is_quit()

  else: 
    print_colors(answer, usr_guess)
    pass





