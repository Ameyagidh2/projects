#include <iostream>
#include<cstdio>
#include<ctime>

using namespace std;

// char board to create a board on the c++ UI
// char board is a 3-D array of 3 rows and 3 columns
char board[3][3] = {{'1','2','3'},{'4','5','6'},{'7','8','9'}};
char current_marker;
int current_player;

void draw_board()
{// function draws the board
    cout<<" "<<board[0][0]<<"|"<<" "<<board[0][1]<<"|"<<" "<<board[0][2]<<endl;
    cout<<"---------\n";
    cout<<" "<<board[1][0]<<"|"<<" "<<board[1][1]<<"|"<<" "<<board[1][2]<<endl;
    cout<<"---------\n";
    cout<<" "<<board[2][0]<<"|"<<" "<<board[2][1]<<"|"<<" "<<board[2][2]<<endl;
    cout<<"\n";
}

bool place_marker(int slot)
{   // function places the marker at current slot position
    int row = (slot / 3);
    int col;
    if (slot % 3 == 0)
    {
        row = row - 1;
        col = 2;
    }
    else{
        col =( slot % 3) - 1;
        // try to find a relationship between
        // suppose eg 7 so between 0,3 and 7
    }
    cout<<"row: "<<row<<" ,"<<"column: "<<col<<endl;
    if(board[row][col] != 'x' && board[row][col] !='o')
    {
      board[row][col] = current_marker;
    return true;
    }
    else{
        return false;
    }
}

void swap_player_and_marker()
{
    if (current_marker == 'x') {
            current_marker = 'o';
    }
    else{
        current_marker = 'x';
    }

    if (current_player == 1 ){
            current_player = 2;
    }
    else{current_player = 1;}
}

int winner()
{
   //Function decides the winner of the game
    for (int i = 0 ; i< 3; i++)
    {   //rows

        if (board[i][0]==board[i][1] && board[i][1] == board[i][2]) return current_player;
            //columns

       if (board[0][i]==board[1][i] && board[1][i] == board[2][i]) return current_player;
    }

     // diagonals
    if (board[0][0]== board[1][1] && board[1][1]== board[2][2]) return current_player;
    if(board[0][2]==board[1][1] && board[1][1] == board[2][0]) return current_player;


    return 0;
}

void game()
{
    cout<<"Game Started!"<<endl;
    cout<<"Player 1 please enter your marker: ";
    char marker_p1;
    cin>>marker_p1;
    current_player = 1;
    current_marker = marker_p1;
    int winner_player;
    for(int j = 0; j<9; j++)
    {
        cout<<"Its "<<current_player<<"'s turn!"<<endl;
        cout<<"Enter your slot"<<endl;
        int slot;
        cin>>slot;
        if(slot<1 || slot>9)
        {
            cout<<"Please enter a valid range for slot";
            j--;
            continue;
        }

        if(!place_marker(slot))
        {
            cout<<"Slot already occupied";
            j--;
            continue;
        }

        draw_board();
        winner_player = winner();
        if(winner_player == 1) {
                cout<<"Player 1 has won the game";
                    break;
                    }
        if(winner_player == 2) {
                cout<<"Player 2 has won the game";
                break;}

        swap_player_and_marker();
    }
    if(winner_player == 0)cout<<"Its a tie"<<endl;
}



int main()
{   cout << "Hello Ameya, Welcome to tic-tac-toe game" << endl;
    draw_board();
    /*
     current_player = 1;
     current_marker = 'X';
     place_marker(5);
     place_marker(2);
     place_marker(8);
    */
    game();

    // for computer to play this game use j / 2 as max not 9
    // [L,U) =(rand() % (U-L+1))+1
    //srand(time(NULL));
    //cout<< (rand() % (10-1+1))+1;

    return 0;
}
