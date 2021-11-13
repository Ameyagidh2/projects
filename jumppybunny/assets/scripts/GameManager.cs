
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//gamemanager to control different states of the gamelevels
/** 1) Menu
 *  2) InGame
 *  3) GameOver
 *  4) Pause
 * 
 * */

public enum GameState
{
    // All stats of the game are in enum which is constant
    Menu,
    InGame,
    GameOver,
    Resume

}
public class GameManager : MonoBehaviour
{
    // Starts our game
    // public object currentGameState of type GameState
    // defalut currentGameState is Menu
    public GameState currentGameState = GameState.Menu;

    // sharedInstance is an object of type GameManager
    // It can be accessed between files
    private static GameManager sharedInstance;
    // this is the sharedInstance which can be accessed by all objects and scripts
    // using the GetInstance function

    private void Awake()
    {
        // called as soon as the object of GameManager called
        //object of gameManager is stored in this shared instance
        sharedInstance = this;
        // this refers to the current object here shared Instance
    }
    public static GameManager GetInstance()
    {
        // static so class method no need to create an object
        // method used to get the private variable of the class
        return sharedInstance;
    }
    public void StartGame()
    {   //controlling the bunning game restart using game manager
        LevelGenerator.sharedInstance.createInitialBlock();
        PlayerController.GetInstance().StartGame();
        ChangeGameState(GameState.InGame);
    }
    private void Start()
    {
        //StartGame();
        // imitially the bunny is at menu position
        currentGameState = GameState.Menu;


    }
    private void Update()
    {
        // when key pressed then game starts
        // To only start game if you are
        // currently not in game
        // currentGameState != GameState.InGame
        if (currentGameState != GameState.InGame && Input.GetButtonDown("s"))
        {
            ChangeGameState(GameState.InGame);
            StartGame();
        }
        if (currentGameState != GameState.InGame && Input.GetKeyDown(KeyCode.S)) 
        {
            ChangeGameState(GameState.InGame);
            StartGame();
        }

    }
    // Called when player dies
    public void GameOver()
    {
        LevelGenerator.sharedInstance.RemoveAllBlocks();
        ChangeGameState(GameState.GameOver);
    }
    // Called when the player decides to quick the game
    // and go to the main menu
    public void BackToMainMenu()
    {
        ChangeGameState(GameState.Menu);
}
    void ChangeGameState(GameState newGameState)
    {
        // this function is used to change current game state
        //changes game state private
        switch (newGameState)
        {
            case GameState.Menu:
                //Let's load Mainmenu Scene
                break;
            case GameState.InGame:
                // Unity Scene must show the Real game
                break;
            case GameState.GameOver:
                // Let's load end of the game scene
                break;
            default:
                newGameState = GameState.Menu;
                break;
        }

        currentGameState = newGameState;
    }
}


/*
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/** 1) Menu
 *  2) InGame
 *  3) GameOver
 *  4) Pause
 * 
 * */
/*
enum DaysOfTheWeek : byte
{
    Monday = 1,
    Tuesday,
    Thursday,
    Friday,
    Saturday,
    Sunday
}
public enum GameState
{
    Menu,
    InGame,
    GameOver,
    Resume
}
public class GameManager : MonoBehaviour
{
    // Starts our game
    // DaysOfTheWeek currentDay = DaysOfTheWeek.Sunday;
    public GameState currentGameState = GameState.Menu;
    private static GameManager sharedInstance;

    private void Awake()
    {
        sharedInstance = this;
    }
    public static GameManager GetInstance()
    {
        return sharedInstance;
    }
    public void StartGame()
    {
        LevelGenerator.sharedInstance.createInitialBlocks();
        PlayerController.GetInstance().StartGame();
        ChangeGameState(GameState.InGame);
    }
    private void Start()
    {
        //StartGame();
        currentGameState = GameState.Menu;
    }
    private void Update()
    {
        if (currentGameState != GameState.InGame && Input.GetButtonDown("s"))
        {
            ChangeGameState(GameState.InGame);
            StartGame();
        }

    }
    // Called when player dies
    public void GameOver()
    {
        LevelGenerator.sharedInstance.RemoveAllBlocks();

        ChangeGameState(GameState.GameOver);
    }
    // Called when the player decides to quick the game
    // and go to the main menu
    public void BackToMainMenu()
    {
        ChangeGameState(GameState.Menu);
    }
    void ChangeGameState(GameState newGameState)
    {
        /* if(newGameState == GameState.Menu)
         {
             //Let's load Mainmenu Scene
         } else if(newGameState == GameState.InGame)
         {
             // Unity Scene must show the Real game
         } else if(newGameState == GameState.GameOver)
         {
             // Let's load end of the game scene
         }
          else{
             currentGameState = GameState.Menu
        }
        */
/*
        switch (newGameState)
        {
            case GameState.Menu:
                //Let's load Mainmenu Scene
                break;
            case GameState.InGame:
                // Unity Scene must show the Real game
                break;
            case GameState.GameOver:
                // Let's load end of the game scene
                break;
            default:
                newGameState = GameState.Menu;
                break;
        }

        currentGameState = newGameState;
    }
}

*/






// Managing stats of our game in unity
// 1)Start
// 2)In Game
// 3)Main Menu
// 4)Pause
// Enum to have all stats and con

// If many gamemanagers then put them in a controller folder but here
// as only one gameManager then put that in the scriptfolder
// Game Manager to change game stats
// Start is called before the first frame update
// Start  of the gam
// Game state type of object named currentGameState and default is Menu

// private statichence should be assessed using a get instance method
// instance to share the game manager among different classes

// at start of the unity game called and storing of object of
// type GameManager

// this refers to the cuurrent object

// method to get access to SharedInstance

// SharedInstance accessed using the methd of GetInstance


//default Menu called at start of game play 

// GameOver called when bunny dies

//public methods to access them outside the main class

// changeGameStaatee
// Game state type is enum
// And newGameState is the state of the current game
// we need to compare the newGame state with the value gamestate.Menu or so
// and then act accorduingly

// Singleton to share an object among all classes

/* if(newGameState == GameState.Menu)
        {
            //Let's load Mainmenu Scene
        } else if(newGameState == GameState.InGame)
        {
            // Unity Scene must show the Real game
        } else if(newGameState == GameState.GameOver)
        {
            // Let's load end of the game scene
        }
         else{
            currentGameState = GameState.Menu
       }
       */