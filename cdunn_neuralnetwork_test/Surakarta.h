/*=====================================================*
名  称：Surakarta.h
说  明：Surakarta游戏规则相关的底层操作
作  者：徐博文
版  本：V1.0.0
更  新：2018.2.17
*======================================================*/

#ifndef _SURAKARTA_H_
#define _SURAKARTA_H_

#define BOARD_WIDTH 6
#define BOARD_HEIGHT 6


/// 名  称：PLAYER
/// 说  明：玩家类型枚举。
/// 作  者：徐博文
/// 时  间：2018.2.16
enum PLAYER
{
	HUMAN = 0,
	MACHINE
};

/// 名  称：EDGEMAP_STRUCT
/// 说  明：边沿映射结构。
/// 作  者：徐博文
/// 时  间：2018.2.16
typedef struct EDGEMAP_STRUCT
{
	int PosX = -1;
	int PosY = -1;
	int PosX2 = -1;
	int PosY2 = -1;
	int stone = -1;
	int stone2 = -1;
}EDGEMAP;

/// 名  称：EDGETYPE
/// 说  明：边沿类型枚举。
/// 作  者：徐博文
/// 时  间：2018.2.16
enum EDGETYPE
{
	LEFTEDGE = 0,
	RIGHTEDGE,
	TOPEDGE,
	BOTTOMEDGE
};

/// 名  称：MOVETYPE
/// 说  明：着法表示类。SrcX、SrcY、TarX、TarY表示一个着法(SrcX,SrcY,TarX,TarY)
/// 作  者：徐博文
/// 时  间：2018.2.16
class MOVETYPE
{
public:
	friend class GAME;

	void Set(int srcx, int srcy, int tarx, int tary);
	void Print();
	bool isEqualTo(MOVETYPE* move);
	MOVETYPE(int srcx, int srcy, int tarx, int tary);

protected:
	int SrcX;
	int SrcY;
	int TarX;
	int TarY;
	bool Attk;
};

/// 名  称：MOVELIST
/// 说  明：着法链表类。储存所有可能的着法。
/// 作  者：徐博文
/// 时  间：2018.2.16
class MOVELIST
{
public:
	friend class MCTS;
	void Add(int srcx, int srcy, int tarx, int tary);
	void Add(MOVELIST* moves);
	void Add(MOVETYPE* move);
	void DelSame(MOVETYPE* move);
	void Del(MOVELIST** new_moves);
	void DelAll(MOVELIST** new_moves);
	bool isEmpty();
	void Print();
	void CopyTo(MOVELIST* moves, bool IdenticalCheck = true);
	MOVELIST();
	~MOVELIST();
private:
	MOVELIST* Last;
	MOVELIST* Next;
	MOVETYPE* Move;
};

/// 名  称：GAME
/// 说  明：游戏类。包含与游戏进程相关的所有属性和函数。
/// 作  者：徐博文
/// 时  间：2018.2.16
class GAME
{
public:
	friend class MCTS;
	GAME();
	GAME(int board[6][6]);
	GAME(int board[6][6], PLAYER player);
	void Reset();
	void PrintPlayer();
	void PrintBoard();
	void PrintEdgemap();
	void Print();
	void SetPlayer(PLAYER player);
	void SetBoard(int board[6][6]);
	MOVELIST* GAME::GetMoves(PLAYER player, int gameBoard[BOARD_HEIGHT][BOARD_WIDTH] = nullptr);
	void Move(MOVETYPE move, int gameBoard[BOARD_HEIGHT][BOARD_WIDTH] = nullptr);
	void Move(int srcx, int srcy, int tarx, int tary, int gameBoard[BOARD_HEIGHT][BOARD_WIDTH] = nullptr);
	inline void ExchangePlayer();
private:
	int Board[BOARD_HEIGHT][BOARD_WIDTH] = { 0 };
	EDGEMAP LeftEdge[6], RightEdge[6], TopEdge[6], BottomEdge[6];
	PLAYER Player = HUMAN;

	MOVETYPE* EdgeMap(EDGETYPE edgetype, int index, int stone);

	const int DefaultBoard[BOARD_HEIGHT][BOARD_WIDTH] = {
		{ 2, 2, 2, 2, 2, 2 },
		{ 2, 2, 2, 2, 2, 2 },
		{ 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0 },
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 1, 1, 1, 1, 1 }
	};
};




#endif // !_SURAKARTA_H_
