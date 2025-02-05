def initialize_board():
    return [' ' for _ in range(9)]  
def print_board(board):
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")
    print("\n")
def check_winner(board, player):
   
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], 
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  
        [0, 4, 8], [2, 4, 6]              
    ]
    
    for combination in win_combinations:
        if board[combination[0]] == board[combination[1]] == board[combination[2]] == player:
            return True
    return False
def minimax(board, depth, is_maximizing, alpha, beta):
   
    if check_winner(board, 'X'):
        return 1 
    if check_winner(board, 'O'):
        return -1  
    if ' ' not in board:
        return 0 

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'  
                eval = minimax(board, depth + 1, False, alpha, beta)
                board[i] = ' '  
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'  
                eval = minimax(board, depth + 1, True, alpha, beta)
                board[i] = ' '  
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  
        return min_eval

def best_move(board):
    best_score = float('-inf')
    move = -1
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'X'  
            score = minimax(board, 0, False, float('-inf'), float('inf'))
            board[i] = ' '  
            if score > best_score:
                best_score = score
                move = i
    return move
def human_move(board):
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if board[move] == ' ':
                board[move] = 'O'  
                break
            else:
                print("The cell is already occupied. Try again.")
        except (ValueError, IndexError):
            print("Invalid move. Please enter a number between 1 and 9.")

def main():
    board = initialize_board()
    print("Welcome to Tic-Tac-Toe!")
    
    while True:
        print_board(board)
        
       
        human_move(board)
        if check_winner(board, 'O'):
            print_board(board)
            print("Congratulations! You win!")
            break
        elif ' ' not in board:
            print_board(board)
            print("It's a draw!")
            break

       
        ai_move = best_move(board)
        board[ai_move] = 'X'
        if check_winner(board, 'X'):
            print_board(board)
            print("AI wins!")
            break
        elif ' ' not in board:
            print_board(board)
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()
    