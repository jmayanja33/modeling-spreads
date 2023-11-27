# Predicting NFL Point Spreads, Totals, and Cover Probabilities (Team 176 Project)

## DESCRIPTION
In this project we created an interactive dashboard on which a user can, for any NFL game, input the set spread and point total for the game from a sports book, and get back the most likely bets that will hit based on those lines (if the favorite or underdog will cover, and the probability the points total is over the set total or not). In addition the user will be presented with predictions on what we think the predicted spread and points total will be.

## INSTALLATION
1.) If not already installed, install Docker Desktop for your operating system (https://docs.docker.com/engine/install/). Open the application once the install is finished, and let it run in the background.

2.) Download the file `docker-compose.yml` from the root folder of this repository.

3.) Open a command prompt and navigate to the folder where `docker-compose.yml` is saved.
- To start the application, run the command: 
```console 
docker compose up -d
```
4.) Once all four containers have been started, open http://localhost:PORT in a browser tab to begin using the dashboard.
- Optionally, you can also open http://localhost:5000/swagger-ui in another browser tab to view the swagger page for the project REST API.

5.) When finished with the application, run the following command in the same command prompt window above to shut down the application:
```console 
docker compose down
```

## EXECUTION

1.) In a browser tab open http://localhost:PORT to view the dashboard.

2.) At the top of the dashboard you will see nine fields for data entry. To get a prediction for a game:

- Input the week the game is to be played, along with the home team and away team.
- Input the current betting lines for the game (favorite team, spread, and points total). Betitng data can easily be found on these sites: 
  - https://sports.yahoo.com/nfl/odds/
  - https://www.espn.com/nfl/lines
- In the final three boxes, input the stadium the game is being played at, if the game is being played at a neutral site, and if the game is a playoff game.

3.) Once all of this data has been inputted click the run box. The resulting display will show which team we think will cover the spread, and give a probability value to represent how confident we are that team will cover. The same will also be true on panels below for the points total. 
As an add on, for anyone interested in betting alternate line, we will provide in the bottom 2 panels predictions on what we think the actual spread of the game will be, as well as how many points will be scored.
