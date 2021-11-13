
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class KillTrigger : MonoBehaviour
{
    private void OnTriggerEnter2D(Collider2D element)
    {
        //on collison with player
        if (element.tag == "Player") {
            // PlayerController is used to set the animation 
            //as it has animator defined there
            //getinstance returns an object
            print("Bunny eliminated! Game Over");
            PlayerController.GetInstance().killPlayer();
        }
    }
}


/*
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class KillTrigger : MonoBehaviour
{
    private void OnTriggerEnter2D(Collider2D element)
    {
        if (element.tag == "Player")
        {
            print("Bunny has been eliminated");
            PlayerController.GetInstance().KillPlayer();
        }
    }
}
*/