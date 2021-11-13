
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

//Class  Function variable is the calling and object is 
//assigned using this method

// through rigid body add force and velocity
public class PlayerController : MonoBehaviour
{   private static PlayerController SharedInstance;
    // shared instance to be accessed outside this class in killTrigger
    private Rigidbody2D rigidBody;
    //rigidBody object
    public float thrust = 10.0f;
    public LayerMask groundLayerMask;
    // layer object
    public Animator animator;
    //animator object
    public float runSpeed = 3.0f;
    // Start is called before the first frame update

    //Initial values of the bunny vector 3d
    //is for position and 2d for velocity

    private Vector3 initialPosition;
    private Vector2 initialVelocity;


    public void Awake()
    {
        SharedInstance = this;
        //rigidBody is the bunny in the scene
        rigidBody = GetComponent<Rigidbody2D>();
        initialVelocity = rigidBody.velocity;
        initialPosition = transform.position;
        animator.SetBool("isAlive", true);
    }
    //Playercontroller type
    public static PlayerController GetInstance()
    {
        return SharedInstance;
    }
    public void StartGame()
    {
        //map the Rigidbody2D component gets the first instance of Rigidbody2d
        //Rigidbody2D compoent is bunny is attached to rigidBody

        //isAlive to keep bunny in frame at game start
        animator.SetBool("isAlive", true);
        // since Gamemanager has a shared object instance
        // this can be performed
        //here class
        // GameManager.GetInstance().BackToMainMenu
        // at start game giving the values of inital position
        transform.position = initialPosition;
        rigidBody.velocity = new Vector2(0, 0);
    }
    private void FixedUpdate()
    {
        //Fixed update is updated every fixed constant time frame
        // precise timem not frame like update method
        GameState currState = GameManager.GetInstance().currentGameState;
        //velocity only when in game
        if (currState == GameState.InGame)
        {
            if (rigidBody.velocity.x < runSpeed)
            {
                //velocity is a vector not a variable
                rigidBody.velocity = new Vector2(runSpeed, 
                    rigidBody.velocity.y);
            }
        }
    }
    // Update is called once per frame
    void Update()
    {
        bool canJump = GameManager.GetInstance().currentGameState == GameState.InGame;
        bool isOnTheGround = IsOnTheGround();
        animator.SetBool("isGrounded", isOnTheGround);
        if (canJump && (Input.GetMouseButtonDown(0)

            || Input.GetKeyDown(KeyCode.Space)
            || Input.GetKeyDown(KeyCode.W)
            ) && isOnTheGround
            )
        {
            Jump();
        }

    }
    void Jump()
    {
        // jumping the bunny which is a rigid body using a force
        rigidBody.AddForce(Vector2.up * thrust, ForceMode2D.Impulse);
    }
    bool IsOnTheGround()
    {
        //Raycast to check collision between objects of the game
        return Physics2D.Raycast(this.transform.position, Vector2.down,
               1.0f, groundLayerMask.value);
        //groundLayerMask is an object hence value is used to get its value
    }
    public void killPlayer()
    { 
    // this is called when trigger is pressed
    animator.SetBool("isAlive", false);
        GameManager.GetInstance().GameOver();
    }
}

/*
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    private Rigidbody2D rigidBody;
    public float thrust = 10.0f;
    public LayerMask groundLayerMask;
    public Animator animator;
    public float runSpeed = 3.0f;
    private static PlayerController sharedInstance;

    private Vector3 initialPosition;
    private Vector2 initialVelocity;

    private void Awake()
    {
        sharedInstance = this;
        rigidBody = GetComponent<Rigidbody2D>();

        initialPosition = transform.position;
        initialVelocity = rigidBody.velocity;
        animator.SetBool("isAlive", true);
    }
    public static PlayerController GetInstance()
    {
        return sharedInstance;
    }
    // Start is called before the first frame update
    public void StartGame()
    {

        animator.SetBool("isAlive", true);
        transform.position = initialPosition;
        rigidBody.velocity = new Vector2(0, 0);


    }
    private void FixedUpdate()
    {
        GameState currState = GameManager.GetInstance().currentGameState;
        if (currState == GameState.InGame)
        {
            if (rigidBody.velocity.x < runSpeed)
            {
                rigidBody.velocity = new Vector2(runSpeed, rigidBody.velocity.y);
            }
        }
    }
    // Update is called once per frame
    void Update()
    {
        bool canJump = GameManager.GetInstance().currentGameState == GameState.InGame;
        bool isOnTheGround = IsOnTheGround();
        animator.SetBool("isGrounded", isOnTheGround);
        if (canJump && (Input.GetMouseButtonDown(0)

            || Input.GetKeyDown(KeyCode.Space)
            || Input.GetKeyDown(KeyCode.W)
            ) && isOnTheGround
            )
        {
            Jump();
        }

    }
    void Jump()
    {

        rigidBody.AddForce(Vector2.up * thrust, ForceMode2D.Impulse);


    }
    bool IsOnTheGround()
    {

        return Physics2D.Raycast(this.transform.position, Vector2.down,
               1.0f, groundLayerMask.value);

    }

    public void KillPlayer()
    {
        animator.SetBool("isAlive", false);
        GameManager.GetInstance().GameOver();

    }
}
*/




















// Animator object to access variables like is alive and grounded
//connnect the spirit to the player

// bunny velocity
// Start is called before the first frame update
// the object is of the type ridig body
// set animator is Alive as true to start with the run animation
// isAlive shows the bunny on the screen


// precise amount  of time this method is calle
// give a constant change in velocity to the bunny
// need to pass vector
// as its not a variable
// increases velocity of the 
// Update is called once per frame

// isOnTheGround is boolean which gets the input from IsOnTheGround
//is activated
// changing the  values of the animator 
// if a key is pressed

// function tells if bunny hit the layer on the ground

//Raycast used to check collision in the scene between objects
//position is initial bunny position
// down is vector direction
// 1.0f is the magnitude of distance threshold
//value used to get int value from the ground layermask 
// Animator to show how animations states are linked to each other