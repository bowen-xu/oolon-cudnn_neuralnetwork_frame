/*=====================================================*
��  �ƣ�Surakarta.h
˵  ����Surakarta��Ϸ������صĵײ����
��  �ߣ��첩��
��  ����V1.0.0
��  �£�2018.2.17
*======================================================*/

#ifndef _SURAKARTA_H_
#define _SURAKARTA_H_

#define BOARD_WIDTH 6
#define BOARD_HEIGHT 6


/// ��  �ƣ�PLAYER
/// ˵  �����������ö�١�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
enum PLAYER
{
	HUMAN = 0,
	MACHINE
};

/// ��  �ƣ�EDGEMAP_STRUCT
/// ˵  ��������ӳ��ṹ��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
typedef struct EDGEMAP_STRUCT
{
	int PosX = -1;
	int PosY = -1;
	int PosX2 = -1;
	int PosY2 = -1;
	int stone = -1;
	int stone2 = -1;
}EDGEMAP;

/// ��  �ƣ�EDGETYPE
/// ˵  ������������ö�١�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
enum EDGETYPE
{
	LEFTEDGE = 0,
	RIGHTEDGE,
	TOPEDGE,
	BOTTOMEDGE
};

/// ��  �ƣ�MOVETYPE
/// ˵  �����ŷ���ʾ�ࡣSrcX��SrcY��TarX��TarY��ʾһ���ŷ�(SrcX,SrcY,TarX,TarY)
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
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

/// ��  �ƣ�MOVELIST
/// ˵  �����ŷ������ࡣ�������п��ܵ��ŷ���
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
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

/// ��  �ƣ�GAME
/// ˵  ������Ϸ�ࡣ��������Ϸ������ص��������Ժͺ�����
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
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
