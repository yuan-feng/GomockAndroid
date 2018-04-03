package yuan.alphazero.gomoku;

import android.util.Log;

import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;
import java.util.Map;
/**
 * Created by yuan on 3/14/18.
 */

public class Board {
    public int[] players = { 0 , 1 } ;
    public HashSet<Integer> availables = new HashSet<Integer>();

    private int width ;
    private int height ;
    private int n_in_row ;

    private Map<Integer,Integer> move2player ;


    private int last_move = -1 ;
    private int current_player ;
    public Board(int width_, int height_, int n_in_row_){

        width = width_ ;
        height = height_ ;
        n_in_row = n_in_row_ ;
        Log.d("my_debug", "Board initialization starting") ;
        for (int i = 0 ; i < width*height; i++ ){
            Integer obj = i ;
            availables.add( obj );
        }
        Log.d("my_debug", "Board initialization ending") ;
    }

    public Tensor<Integer> current_state(){
        int[][][] board_states = new int[4][width][height];
        for( Map.Entry<Integer,Integer> entry: move2player.entrySet() ){
            int move = entry.getKey() ;
            int player = entry.getValue() ;
            board_states[player][move/width][move%width] = 1 ;
            board_states[2][last_move/width][last_move%width] = 1 ;
        }
        if( move2player.size() % 2 == 0 ){
            for (int i = 0 ; i < width ; i++ ){
                for (int j = 0 ; j < height; j++ ){
                    board_states[3][i][j] = 1 ;
                }
            }
        }
        Tensor<Integer> board_state_t = Tensor.create( board_states, int.class) ;
        return board_state_t;
    }

    public void do_move(int move){
        move2player.put(move, current_player) ;
        availables.remove(move) ;
        last_move = move ;
        current_player = 1 - current_player;
    }

    public void do_move(int xMove, int yMove){
        int[] loc = {xMove, yMove};
        int move = location_to_move(loc);
        do_move(move);
    }









    public int[] move_to_location(int move){
        int[] loc = new int[2];
        loc[0] = move / width ;
        loc[1] = move % width ;
        return loc ;
    }

    public int location_to_move(int[] loc){
        int h = loc[0] ;
        int w = loc[1] ;
        int move = h * width + w ;
        return move ;
    }
}
