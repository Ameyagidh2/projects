
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LevelGenerator : MonoBehaviour
{
    // LevelGenerator gives output as blocks from
    //LevelBlock class

    // list of type Level block class
    //sample Blocks from where to create new blocks
    public List<LevelBlock> legoBlocks = new List<LevelBlock>();
    //list of type level block

    //Blocks added to the game
    List<LevelBlock> currentBlocks = new List<LevelBlock>();
    public Transform initialPoint;
    private static LevelGenerator _sharedInstance;
    public byte initialBlockNumber = 2;
    public static LevelGenerator sharedInstance
    {
        get {
            return _sharedInstance;
        }
    }
    private void Awake()
    {
        _sharedInstance = this;
        createInitialBlock();
       
    }
    public void createInitialBlock()
       {
        if (currentBlocks.Count > 0)
        {
            return;
        }
        for (byte i = 0; i < initialBlockNumber; i++)
        {
            AddNewBlock(true);
            // true to check initial block or not
        }
    }
    
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }
    public void AddNewBlock(bool initialBlocks = false)
    {
        //method used to add new blocks randomly for levels
        //legoBlock is a list of type level block
        //Generating random number for entry to legoblocks

        int randomNumber = initialBlocks ? 0 : Random.Range(0,legoBlocks.Count);
    //instantiate gives object of class LevelBlock
    // block has object of type level block which
    // is used to generate new bl
        var block = Instantiate(legoBlocks[randomNumber]);
        //block will have a random legoblock which
        //is object of type LevelBlock

        //to add new blocks to levelgenerator class as parent 
        //transform used to get access object
        block.transform.SetParent(this.transform);
        //this is the levelGenerator object
        //Initial position
        Vector3 blockPosition = Vector3.zero;
        //adding block how
        if (currentBlocks.Count == 0)
        {
            blockPosition = initialPoint.position;
            //initialPoint is an empty object hence postion 
            //can be used
        }
        else {
            //set last block position
            int lastBlockpos = currentBlocks.Count - 1;
            blockPosition = currentBlocks[lastBlockpos].exitPoint.position;
        }
        block.transform.position = blockPosition;
        currentBlocks.Add(block);
    }
    public void RemoveOldBlock() {

        var oldblock = currentBlocks[0];
        currentBlocks.Remove(oldblock);
        //destroying game object
        Destroy(oldblock.gameObject);
    }
    public void RemoveAllBlocks()
    {
        // removes all blocks
        while (currentBlocks.Count >0 )
        {
            RemoveOldBlock();
        }
    }
}

