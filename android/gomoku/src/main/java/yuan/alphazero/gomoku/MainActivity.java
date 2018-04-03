package yuan.alphazero.gomoku ;

import android.app.Activity;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.content.Context;
import android.content.res.Resources;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class MainActivity extends Activity {
    // declare a int for board size
    final static int maxN = 6;
    final static int n_in_row = 4;
    private Context context;
    // declare for imageView (Cells) array
    private ImageView[][] ivCell = new ImageView[maxN][maxN];

    private Drawable[] drawCell = new Drawable[4] ;

    private Button btnPlay;
    private TextView tvTurn ;

    private int[][] valueCell = new int[maxN][maxN] ; // 0 is empty, 1 is player, 2 is bot
    private int winner_play; //who is winne? 0 is none, 1 is player, 2 is bot.
    private boolean firstMove ;
    private int xMove, yMove; // x and y axis of cell ==> define position of cell.
    private int turnPlay ; // whose turn ?
    private boolean isClicked ; // track player click cel or not => make sure that player only cleck to 1 cell.

    private Board board = new Board(maxN, maxN, n_in_row);
    private PolicyValueNet nNet = new PolicyValueNet(maxN, maxN) ;
    public MainActivity(){

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        context=this;
        setListen();
        loadResources();
        designBoardGame();
//        nNet.restore_neural_network();
    }

    private void setListen(){
        btnPlay = (Button) findViewById(R.id.btnPlay) ;
        tvTurn = (TextView) findViewById( R.id.tvTurn );
        btnPlay.setText("Play Game");;
        tvTurn.setText("Press button Play Game");

        btnPlay.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                init_game();
                play_game();
            }
        });
    }

    private void play_game() {
        // we need define who play first
        Random r = new Random() ;
        turnPlay = r.nextInt(2) + 1 ;// r.nextint(2) return value in (0,1)

        if (turnPlay == 1){ // player play first
            // inform => make a toast
            Toast.makeText(context, "Player play first!", Toast.LENGTH_SHORT).show() ;
            playerTurn();
        }else{ // bot turn
            Toast.makeText(context, "Bot play first!", Toast.LENGTH_SHORT).show();
            botTurn();
        }
    }

    private void botTurn() {
        tvTurn.setText("Bot");
        // if this is first move, bot always choose center cell (7,7)
        if( firstMove ){
            firstMove = false ;
            xMove = 7 ; yMove = 7 ;
            make_a_move() ;

        }else{
            // try to find best xMove, yMove
            findBotMove();
            make_a_move();
        }
    }

    private final int [] iRow = {-1,-1,-1, 0, 1, 1, 1, 0} ;
    private final int [] iCol = {-1, 0, 1, 1, 1, 0,-1,-1} ;
    private void findBotMove() {

        nNet.feed(board.current_state());
        int[] MoveLoc = nNet.next_move();

        xMove = MoveLoc[0] ;
        yMove = MoveLoc[1] ;
        board.do_move(xMove, yMove);
    }



    private void make_a_move() {
//        Log.d("debuggg", "Make a move with ("+xMove+", " + yMove+"). Turn: "+ turnPlay) ;
        ivCell[xMove][yMove].setImageDrawable(drawCell[turnPlay]);
        valueCell[xMove][yMove] = turnPlay ;
        // check if anyone win
        // if no empty cell exist=> draw
        if(noEmptyCell()){
            Toast.makeText(context, "Draw!!!", Toast.LENGTH_SHORT).show() ;
            return;
        }else{
            if(CheckWinner()){
                if (winner_play == 1 ){
                    Toast.makeText(context, "Winner is Player", Toast.LENGTH_SHORT).show();
                    tvTurn.setText("Winner is player");
                }else{
                    Toast.makeText(context, "Winner is Bot ", Toast.LENGTH_SHORT).show();
                    tvTurn.setText("Winner is bot");
                }
                return;
            }
        }

        if(turnPlay==1){ // player
            turnPlay = 3 - turnPlay ;
//            Log.d("debuggg", "botTurn") ;
            botTurn();
        }else{
            turnPlay = 3 - turnPlay ;
//            Log.d("debuggg", "playerTurn") ;
            playerTurn();
        }
    }

    private boolean CheckWinner() {
        // we only need to check the recent xMove, yMove can create 5 cells in a rowor not
        if( winner_play != 0 ){
            return true;
        }
        // check in row
        VectorEnd(xMove,0, 0, 1, xMove, yMove);
        // check in column
        VectorEnd(0,yMove, 1, 0, xMove, yMove);
        // check left to right
        if (xMove + yMove >= maxN -1 ){
            VectorEnd(maxN-1, xMove+yMove-maxN+1, -1,1,xMove,yMove);
        }else{
            VectorEnd(xMove+yMove, 0, -1,1,xMove,yMove);
        }
        if (xMove <= yMove){
            VectorEnd(xMove-yMove+maxN-1, maxN-1, -1,-1, xMove, yMove);
        }else{
            VectorEnd(maxN-1, maxN-1-(xMove-yMove), -1,-1, xMove, yMove);
        }
        if (winner_play != 0 ){return true;}
        return false;

    }

    private void VectorEnd(int xx, int yy, int vx, int vy, int rx, int ry) {
        // this void will check the row base on vector (vx, vy) in range
        // (rx,ry) - 4*(vx,vy) =>(rx,ry)+4*(vx,vy)
        if (winner_play !=0 ){return;}
        final int range = 4 ;
        int i, j ;
        int xbelow = rx - range * vx ;
        int ybelow = ry - range * vy ;
        int xabove = rx + range * vx ;
        int yabove = ry + range * vy ;
        String st = "" ;
        i = xx; j = yy ;
        while( !inside(i,xbelow, xabove) || !inside(j,ybelow, yabove) ){
            i += vx ; j+= vy ;
        }
        while( true ){
            st = st + String.valueOf(valueCell[i][j]) ;
            if ( st.length() == 5 ){
                EvalEnd(st);
                st = st.substring(1,5) ; // substring of st from index 1->5 => delete first character
            }
            i += vx ; j += vy ;
            if( !inBoard(i,j) || !inside(i,xbelow, xabove) || !inside(j,ybelow, yabove)
                    || winner_play != 0 ){
                break;
            }
        }
    }

    private boolean inBoard(int i, int j) {
        // check i,j in board or not
        if ( i < 0 || i > maxN - 1 ||  j<0 || j>maxN -1 ) {
            return false;
        }
        return true;
    }

    private void EvalEnd(String st) {
        switch (st){
            case "11111": winner_play=1; break;
            case "22222": winner_play=2; break;
            default:break;
        }
    }

    private boolean inside(int i, int xbelow, int xabove) {// this check i is (xbelow,xabove) or not
        return (i-xbelow) * (i-xabove) <= 0 ;

    }

    private boolean noEmptyCell() {
        for (int i = 0 ; i<maxN; i++ ){
            for (int j = 0 ; j < maxN ; j++ ){
                if( valueCell[i][j] == 0 ){
                    return false;
                }
            }
        }
        return true;
    }

    private void playerTurn() {
        Log.d("debuggg", "player turn");
        tvTurn.setText("Player");
        // we get xMove, yMove of Player by the way listen click on cell.
        // so turn listen on
        firstMove = false ;
        isClicked = false ;

    }

    private void init_game() {
        // this void will create UI before game start
        // for game control, we need some variables.
        firstMove = true ;
        winner_play = 0 ;
        for ( int i = 0 ; i< maxN ; i++ ){
            for (int j = 0 ; j < maxN ; j++ ){
                ivCell[i][j].setImageDrawable(drawCell[0]); // default or empty cell
                valueCell[i][j] = 0 ;
            }
        }
        /////////// above is init for game
    }


    private void loadResources(){
        drawCell[3] = context.getResources().getDrawable(R.drawable.cell_bg); // background.
        // copy 2 images for 2 drawable player and bot
        // edit it
        drawCell[0] = null ;
        drawCell[1] = context.getResources().getDrawable(R.drawable.blue);
        drawCell[2] = context.getResources().getDrawable(R.drawable.green);

    }

    private void designBoardGame(){
        // create layour parans to optimize size of cell
        // we create a horizontal linearlayout for a row
        // which contains maxN imageView in
        // new to find out size of cell first

        int sizeofCell = Math.round(ScreenWidth()/maxN);
        LinearLayout.LayoutParams lpRow = new LinearLayout.LayoutParams(sizeofCell*maxN, sizeofCell);
        LinearLayout.LayoutParams lpCell = new LinearLayout.LayoutParams(sizeofCell, sizeofCell);

        LinearLayout linBoardGame = findViewById(R.id.linBoardGame) ;

        // create cells
        for(int i = 0 ; i < maxN ; i++ ){
            LinearLayout linRow = new LinearLayout(context);
            // make a row
            for (int j = 0 ; j < maxN ; j++ ){
                ivCell[i][j] = new ImageView(context);
                // make a cell
                // need to set background default for cell
                // cell has 3 states, empty (default), player, bot
                ivCell[i][j].setBackground(drawCell[3]);
                final int x = i ;
                final int y = j ;
                // make that for safe and clear ;
                ivCell[i][j].setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        if( valueCell[x][y] == 0 ){
                            if ( turnPlay == 1 || !isClicked ){ // turn of player
                                isClicked = true;
                                xMove = x ; yMove = y ; // i,j must be final variables.
                                board.do_move(xMove,yMove);
                                make_a_move();
                            }
                        }
                    }
                });
                linRow.addView(ivCell[i][j], lpCell);
            }
            linBoardGame.addView(linRow, lpRow);
        }
    }

    private float ScreenWidth(){
        Resources resources = context.getResources() ;
        DisplayMetrics dm = resources.getDisplayMetrics();
        return dm.widthPixels ;
    }

}
