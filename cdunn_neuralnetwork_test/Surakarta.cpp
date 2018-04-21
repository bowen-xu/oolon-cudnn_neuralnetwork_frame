/*=====================================================*
��  �ƣ�Surakarta.cpp
˵  ����Surakarta��Ϸ������صĵײ����
��  �ߣ��첩��
��  ����V1.0.0
��  �£�2018.2.17
*======================================================*/

#include "Surakarta.h"
#include <stdio.h>


/// ��������Set
/// ��  ����int srcx, int srcy, int tarx, int tary
/// ˵  �������ŷ�����Ϊ(srcx,srcy,tarx,tary)��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVETYPE::Set(int srcx, int srcy, int tarx, int tary)
{
	SrcX = srcx;
	SrcY = srcy;
	TarX = tarx;
	TarY = tary;
}

/// ��������MOVETYPE
/// ��  ����int srcx, int srcy, int tarx, int tary
/// ˵  ������MOVETYPE�Ĺ��캯�������ŷ�����Ϊ(srcx,srcy,tarx,tary)��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
MOVETYPE::MOVETYPE(int srcx, int srcy, int tarx, int tary)
{
	Set(srcx, srcy, tarx, tary);
}

/// ��������isEqualTo
/// ��  ����MOVETYPE* move
/// ˵  �����жϵ�ǰ�ŷ������move�ŷ��Ƿ���ͬ������ͬ�򷵻�true��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.17
bool MOVETYPE::isEqualTo(MOVETYPE* move)
{
	if (move->SrcX == SrcX && move->SrcY == SrcY && move->TarX == TarX && move->TarY == TarY)
	{
		return true;
	}
	else
	{
		return false;
	}
}


/// ��������Print
/// ��  ����none
/// ˵  ������ӡ�ŷ���
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVETYPE::Print()
{
	printf("(%d,%d,%d,%d)", SrcX, SrcY, TarX, TarY);
}

/// ��������isEmpty
/// ��  ����none
/// ˵  �����жϿ�ѡ�ŷ��Ƿ�Ϊ�ա�����true��Ϊ�ա�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
bool MOVELIST::isEmpty()
{
	if (Move == nullptr)
	{
		return true;
	}
	else
	{
		return false;
	}
}


/// ��������MOVELIST
/// ��  ����none
/// ˵  ������MOVELIST�Ĺ��캯����
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
MOVELIST::MOVELIST()
{
	Last = nullptr;
	Next = nullptr;
	Move = nullptr;
}

/// ��������Add
/// ��  ����int srcx, int srcy, int tarx, int tary
/// ˵  �������ѡ�ŷ����������һ���ŷ�(srcx,srcy,tarx,tary)��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::Add(int srcx, int srcy, int tarx, int tary)
{
	MOVELIST* newmovelist;
	MOVETYPE* newmove;

	newmove = new(MOVETYPE)(srcx, srcy, tarx, tary);

	if (Move == nullptr)
	{
		Move = newmove;
	}
	else
	{
		newmovelist = new(MOVELIST);
		newmovelist->Move = newmove;
		if (Next != nullptr)
		{
			Next->Last = newmovelist;
			//newmovelist->Next = Next;
		}
		newmovelist->Next = Next;
		Next = newmovelist;
		newmovelist->Last = this;

	}
}

/// ��������Add
/// ��  ����MOVETYPE* move
/// ˵  �������ѡ�ŷ����������һ���ŷ�(srcx,srcy,tarx,tary)
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::Add(MOVETYPE* move)
{
	MOVELIST* newmovelist;

	if (Move == nullptr)
	{
		Move = move;
	}
	else
	{
		newmovelist = new(MOVELIST);
		newmovelist->Move = move;
		if (Next != nullptr)
		{
			Next->Last = newmovelist;
		}
		newmovelist->Next = Next;
		Next = newmovelist;
		newmovelist->Last = this;
	}
}

/// ��������Add
/// ��  ����MOVELIST* moves
/// ˵  �������ѡ�ŷ����������һ���ŷ�(srcx,srcy,tarx,tary)
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::Add(MOVELIST* moves)
{
	if (Move == nullptr)
	{
		Move = moves->Move;
		moves->Move = nullptr;
		moves->Del(&moves);
	}
	else
	{
		if (Next != nullptr)
		{
			Next->Last = moves;
		}
		moves->Next = Next;
		Next = moves;
		moves->Last = this;
	}
}

/// ��������DelAll
/// ��  ����MOVELIST** new_moves = nullptr
/// ˵  ������տ�ѡ�ŷ���������new_moves��������Ϊ��ָ�룬һ��ɴ����ŷ�����ı�ͷ��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::DelAll(MOVELIST** new_moves = nullptr)
{
	if (new_moves != nullptr)
	{
		*new_moves = nullptr;
	}
	if (Last != nullptr)
	{
		Last->Next = nullptr;
		Last->DelAll();
	}
	if (Next != nullptr)
	{
		Next->Last = nullptr;
		Next->DelAll();
	}
	
	delete(this);
}

/// ��������Del
/// ��  ����MOVELIST** new_moves
/// ˵  ����ɾ����ѡ�ŷ�������new_movesָ���һ���ŷ���
///			new_moves��������Ϊ��ɾ�ŷ�����һ���ŷ���
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::Del(MOVELIST** new_moves)
{
	*new_moves = Next;

	if (Last != nullptr)
	{
		Last->Next = Next;
	}

	if (Next != nullptr)
	{
		Next->Last = Last;
	}

	//Last = nullptr;
	//Next = nullptr;

	delete(this);
}

/// ��������~MOVELIST
/// ��  ����none
/// ˵  ������MOVELIST�������������ͷ��ŷ���ռ���ڴ�ռ䡣
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
MOVELIST::~MOVELIST()
{
	delete(Move);
}

/// ��������Print
/// ��  ����none
/// ˵  ������ӡ�ŷ������е������ŷ���
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::Print()
{
	MOVELIST* moves;
	printf("\nMoves:\n");
	printf("(SrcX,SrcY,TarX,TarY):\t");
	moves = this;
	if (moves == nullptr)
	{
		printf("null\n");
		return;
	}
	do {
		moves->Move->Print();
		printf("\t");
		moves = moves->Next;
	} while (moves != nullptr);
	
}

/// ��������DelSame
/// ��  ����MOVETYPE* move
/// ˵  ����ɾ���ŷ�����ĵ�ǰ�ŷ����Ժ�������ŷ��У������move�ŷ���ͬ�������ŷ���
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::DelSame(MOVETYPE* move)
{
	MOVELIST* moves_buff;

	moves_buff = this;

	while (moves_buff != nullptr)
	{
		if (moves_buff->Move->isEqualTo(move))
		{
			moves_buff->Del(&moves_buff);
		}
		else
		{
			moves_buff = moves_buff->Next;
		}
	}
}

/// ��������CopyTo
/// ��  ����MOVELIST* moves, bool IdenticalCheck
/// ˵  ����������moves�ŷ������е�һ������������ǰ�ŷ������У�������ԭmoves�����Ƴ���
///			����IdenticalCheck���ڼ����ͬ���ŷ�����IdenticalCheckΪtrue�����ڿ����ŷ�
///			ǰ����һ���ظ��Լ�飬����moves����������ŷ���ͬ�������ŷ���ԭmoves����ɾ����
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void MOVELIST::CopyTo(MOVELIST* moves, bool IdenticalCheck)
{
	MOVELIST* moves_buf;
	MOVELIST* moves_attack;

	moves_buf = this;
	moves_attack = this;
	while (moves_buf != nullptr)
	{
		if (IdenticalCheck == true)
		{
			moves_buf->Next->DelSame(moves_buf->Move);
		}
		moves_buf = moves_buf->Next;
		moves->Add(moves_attack);
		moves_attack = moves_buf;
	}
}

/// ��������GAME
/// ��  ����int board[6][6]
/// ˵  ������GAME�Ĺ��캯������board����Ϣ���������̡�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
GAME::GAME(int board[6][6])
{
	SetBoard(board);
}


/// ��������GAME
/// ��  ����int board[6][6], PLAYER player
/// ˵  ������GAME�Ĺ��캯������board����Ϣ���������̣������������Ϊplayer��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
GAME::GAME(int board[6][6], PLAYER player)
{
	SetBoard(board);
	SetPlayer(player);
}

/// ��������GAME
/// ��  ����none
/// ˵  ������GAME�Ĺ��캯�����������̡�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
GAME::GAME()
{
	Reset();
}


/// ��������Move
/// ��  ����MOVETYPE move
/// ˵  �������ӣ�������������Ϣ���ŷ�ΪMove(srcx,srcy.tarx,tary)
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::Move(MOVETYPE move, int gameBoard[BOARD_HEIGHT][BOARD_WIDTH])
{
	//Board[move.TarY][move.TarX] = Board[move.SrcY][move.SrcX];
	//Board[move.SrcY][move.SrcX] = 0;
	Move(move.SrcX,move.SrcY,move.TarX,move.TarY, gameBoard);
}

/// ��������Move
/// ��  ����int srcx, int srcy, int tarx, int tary
/// ˵  �������ӣ�������������Ϣ���ŷ�Ϊ(srcx,srcy.tarx,tary)
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::Move(int srcx, int srcy, int tarx, int tary, int gameBoard[BOARD_HEIGHT][BOARD_WIDTH])
{
	if (gameBoard == nullptr)
	{
		gameBoard = Board;
		ExchangePlayer();

	}
	gameBoard[tary][tarx] = gameBoard[srcy][srcx];
	gameBoard[srcy][srcx] = 0;
	//Move(MOVETYPE(srcx, srcy, tarx, tary));
}



inline void GAME::ExchangePlayer()
{
	if (Player == HUMAN)
	{
		Player = MACHINE;
	}
	else
	{
		Player = HUMAN;
	}
}



/// ��������GetMoves
/// ��  ����PLAYER player
/// ˵  ������ȡ��ǰ�����п��ܵ��ŷ�������playerΪ��ǰ���ӵ���ҡ�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
MOVELIST* GAME::GetMoves(PLAYER player, int gameBoard[BOARD_HEIGHT][BOARD_WIDTH])
{
	int stone;		// ��������
	int stone_opp;	// �Է�����
	int stone_edge;

	MOVELIST* moves;
	MOVETYPE* newmove;
	MOVELIST* moves_attack;
	MOVETYPE* move_attack;


	moves = new(MOVELIST);
	moves_attack = new(MOVELIST);
	move_attack = nullptr;

	if (gameBoard == nullptr)
	{
		gameBoard = Board;
	}

	if (player == HUMAN)
	{
		stone = 1;
		stone_opp = 2;
	}
	else if (player == MACHINE)
	{
		stone = 2;
		stone_opp = 1;
	}

	// 1. �ҳ��������� �� �ƶ��ŷ�
	for (int i = 0; i < BOARD_WIDTH; i++)
	{
		for (int j = 0; j < BOARD_HEIGHT; j++)
		{
			if (gameBoard[i][j] != 0)
			{
#pragma region ��������
				if (LeftEdge[i].PosX < 0)
				{
					LeftEdge[i].PosX = j;
					LeftEdge[i].PosY = i;
					LeftEdge[i].stone = gameBoard[i][j];
				}
				else if (LeftEdge[i].PosX2 < 0)
				{
					LeftEdge[i].PosX2 = j;
					LeftEdge[i].PosY2 = i;
					LeftEdge[i].stone2 = gameBoard[i][j];
				}
				if (RightEdge[i].PosX < j)
				{
					RightEdge[i].PosX2 = RightEdge[i].PosX;
					RightEdge[i].PosY2 = RightEdge[i].PosY;
					RightEdge[i].stone2 = RightEdge[i].stone;
					RightEdge[i].PosX = j;
					RightEdge[i].PosY = i;
					RightEdge[i].stone = gameBoard[i][j];
				}
				if (TopEdge[j].PosY < 0)
				{
					TopEdge[j].PosX = j;
					TopEdge[j].PosY = i;
					TopEdge[j].stone = gameBoard[i][j];
				}
				else if (TopEdge[j].PosY2 < 0)
				{
					TopEdge[j].PosX2 = j;
					TopEdge[j].PosY2 = i;
					TopEdge[j].stone2 = gameBoard[i][j];
				}
				if (BottomEdge[j].PosY < i)
				{
					BottomEdge[j].PosX2 = BottomEdge[j].PosX;
					BottomEdge[j].PosY2 = BottomEdge[j].PosY;
					BottomEdge[j].stone2 = BottomEdge[j].stone;
					BottomEdge[j].PosX = j;
					BottomEdge[j].PosY = i;
					BottomEdge[j].stone = gameBoard[i][j];
				}
#pragma endregion
#pragma region �ƶ��ŷ�
				if (gameBoard[i][j] == stone)
				{
					if (i > 0)
					{
						if (gameBoard[i - 1][j] == 0)
						{
							moves->Add(j, i, j, i - 1);
						}

						if (j > 0)
						{
							if (gameBoard[i - 1][j - 1] == 0)
							{
								moves->Add(j, i, j - 1, i - 1);
							}
						}
						if (j < 5)
						{
							if (gameBoard[i - 1][j + 1] == 0)
							{
								moves->Add(j, i, j + 1, i - 1);
							}
						}
					}
					if (i < 5)
					{
						if (gameBoard[i + 1][j] == 0)
						{
							moves->Add(j, i, j, i + 1);
						}

						if (j > 0)
						{
							if (gameBoard[i + 1][j - 1] == 0)
							{
								moves->Add(j, i, j - 1, i + 1);
							}
						}
						if (j < 5)
						{
							if (gameBoard[i + 1][j + 1] == 0)
							{
								moves->Add(j, i, j + 1, i + 1);
							}
						}
					}

					if (j > 0)
					{
						if (gameBoard[i][j - 1] == 0)
						{
							moves->Add(j, i, j - 1, i);
						}
					}
					if (j < 5)
					{
						if (gameBoard[i][j + 1] == 0)
						{
							moves->Add(j, i, j + 1, i);
						}
					}
				}
#pragma endregion
			}
		}
	}

	// 2. �ҳ����г����ŷ�
	for (int i = 1; i <= 4; i++)
	{

		stone_edge = LeftEdge[i].stone;
		if (stone_edge == stone)
		{
			move_attack = EdgeMap(LEFTEDGE, i, stone_edge);
			if (move_attack != nullptr)
			{
				moves_attack->Add(move_attack);
			}
		}
		stone_edge = TopEdge[i].stone;
		if (stone_edge == stone)
		{
			move_attack = EdgeMap(TOPEDGE, i, stone_edge);
			if (move_attack != nullptr)
			{
				moves_attack->Add(move_attack);
			}
		}
		stone_edge = RightEdge[i].stone;
		if (stone_edge == stone)
		{
			move_attack = EdgeMap(RIGHTEDGE, i, stone_edge);
			if (move_attack != nullptr)
			{
				moves_attack->Add(move_attack);
			}
		}
		stone_edge = BottomEdge[i].stone;
		if (stone_edge == stone)
		{
			move_attack = EdgeMap(BOTTOMEDGE, i, stone_edge);
			if (move_attack != nullptr)
			{
				moves_attack->Add(move_attack);
			}
		}
	}

	// 3. ȥ���ظ������ŷ�
	if (moves_attack->isEmpty())
	{
		moves_attack->DelAll();
		moves_attack = nullptr;
	}
	else
	{
		moves_attack->CopyTo(moves, true);
	}

	if (moves->isEmpty())
	{
		moves->DelAll();
		moves = nullptr;
	}

	return moves;

}


/// ��������EdgeMap
/// ��  ����EDGETYPE edgetype, int index, int stone
/// ˵  ������ȡ����ӳ�䡣����ֵΪӳ�������������ŷ���
///			����edgetypeΪ�������ͣ�indexΪ��ǰ����������stoneΪ��ӳ������ӡ�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
MOVETYPE* GAME::EdgeMap(EDGETYPE edgetype, int index, int stone)
{
	MOVETYPE* move;

	move = nullptr;

	if (edgetype == LEFTEDGE)
	{
		if (index > 0 && index <= 2)
		{
			if (TopEdge[index].stone == -1)
			{
				move = EdgeMap(BOTTOMEDGE, index, stone);
				if (move != nullptr)
				{
					move->Set(LeftEdge[index].PosX, LeftEdge[index].PosY, move->TarX, move->TarY);
				}
			}
			else if (TopEdge[index].stone == stone)
			{
				if (TopEdge[index].PosX == LeftEdge[index].PosX && TopEdge[index].PosY == LeftEdge[index].PosY)
				{
					if (TopEdge[index].stone2 == -1)
					{
						move = EdgeMap(BOTTOMEDGE, index, stone);
						if (move != nullptr)
						{
							move->Set(LeftEdge[index].PosX, LeftEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (TopEdge[index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(LeftEdge[index].PosX, LeftEdge[index].PosY, TopEdge[index].PosX2, TopEdge[index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(LeftEdge[index].PosX, LeftEdge[index].PosY, TopEdge[index].PosX, TopEdge[index].PosY);
			}
		}
		else if (index > 2 && index <= 4)
		{
			if (BottomEdge[BOARD_WIDTH - 1 - index].stone == -1)
			{
				move = EdgeMap(TOPEDGE, BOARD_WIDTH - 1 - index, stone);
				if (move != nullptr)
				{
					move->Set(LeftEdge[index].PosX, LeftEdge[index].PosY, move->TarX, move->TarY);
				}
			}
			else if (BottomEdge[BOARD_WIDTH - 1 - index].stone == stone)
			{
				if (BottomEdge[BOARD_WIDTH - 1 - index].PosX == LeftEdge[index].PosX && BottomEdge[BOARD_WIDTH - 1 - index].PosY == LeftEdge[index].PosY)
				{
					if (BottomEdge[BOARD_WIDTH - 1 - index].stone2 == -1)
					{
						move = EdgeMap(TOPEDGE, BOARD_WIDTH - 1 - index, stone);
						if (move != nullptr)
						{
							move->Set(LeftEdge[index].PosX, LeftEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (BottomEdge[BOARD_WIDTH - 1 - index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(LeftEdge[index].PosX, LeftEdge[index].PosY, BottomEdge[BOARD_WIDTH - 1 - index].PosX2, BottomEdge[BOARD_WIDTH - 1 - index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(LeftEdge[index].PosX, LeftEdge[index].PosY, BottomEdge[BOARD_WIDTH - 1 - index].PosX, BottomEdge[BOARD_WIDTH - 1 - index].PosY);
			}
		}
	}
	else if (edgetype == RIGHTEDGE)
	{
		if (index > 0 && index <= 2)
		{
			if (TopEdge[BOARD_WIDTH - 1 - index].stone == -1)
			{
				move = EdgeMap(BOTTOMEDGE, BOARD_WIDTH - 1 - index, stone);
				if (move != nullptr)
				{
					move->Set(RightEdge[index].PosX, RightEdge[index].PosY, move->TarX, move->TarY);
				}
			}
			else if (TopEdge[BOARD_WIDTH - 1 - index].stone == stone)
			{
				if (TopEdge[BOARD_WIDTH - 1 - index].PosX == RightEdge[index].PosX && TopEdge[BOARD_WIDTH - 1 - index].PosY == RightEdge[index].PosY)
				{
					if (TopEdge[BOARD_WIDTH - 1 - index].stone2 == -1)
					{
						move = EdgeMap(BOTTOMEDGE, BOARD_WIDTH - 1 - index, stone);
						if (move != nullptr)
						{
							move->Set(RightEdge[index].PosX, RightEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (TopEdge[BOARD_WIDTH - 1 - index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(RightEdge[index].PosX, RightEdge[index].PosY, TopEdge[BOARD_WIDTH - 1 - index].PosX2, TopEdge[BOARD_WIDTH - 1 - index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(RightEdge[index].PosX, RightEdge[index].PosY, TopEdge[BOARD_WIDTH - 1 - index].PosX, TopEdge[BOARD_WIDTH - 1 - index].PosY);
			}
		}
		else if (index > 2 && index <= 4)
		{
			if (BottomEdge[index].stone == -1)
			{
				move = EdgeMap(TOPEDGE, index, stone);
				if (move != nullptr)
				{
					move->Set(RightEdge[index].PosX, RightEdge[index].PosY, move->TarX, move->TarY);
				}
			}
			else if (BottomEdge[index].stone == stone)
			{
				if (BottomEdge[index].PosX == RightEdge[index].PosX && BottomEdge[index].PosY == RightEdge[index].PosY)
				{
					if (BottomEdge[index].stone2 == -1)
					{
						move = EdgeMap(TOPEDGE, index, stone);
						if (move != nullptr)
						{
							move->Set(RightEdge[index].PosX, RightEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (BottomEdge[index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(RightEdge[index].PosX, RightEdge[index].PosY, BottomEdge[index].PosX2, BottomEdge[index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(RightEdge[index].PosX, RightEdge[index].PosY, BottomEdge[index].PosX, BottomEdge[index].PosY);
			}
		}
	}
	else if (edgetype == TOPEDGE)
	{
		if (index > 0 && index <= 2)
		{
			if (LeftEdge[index].stone == -1)
			{
				move = EdgeMap(RIGHTEDGE, index, stone);
				if (move != nullptr)
				{
					move->Set(TopEdge[index].PosX, TopEdge[index].PosY, move->TarX, move->TarY);
				}
			}
			else if (LeftEdge[index].stone == stone)
			{
				if (LeftEdge[index].PosX == TopEdge[index].PosX && LeftEdge[index].PosY == TopEdge[index].PosY)
				{
					if (LeftEdge[index].stone2 == -1)
					{
						move = EdgeMap(RIGHTEDGE, index, stone);
						if (move != nullptr)
						{
							move->Set(TopEdge[index].PosX, TopEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (LeftEdge[index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(TopEdge[index].PosX, TopEdge[index].PosY, LeftEdge[index].PosX2, LeftEdge[index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(TopEdge[index].PosX, TopEdge[index].PosY, LeftEdge[index].PosX, LeftEdge[index].PosY);
			}
		}
		else if (index > 2 && index <= 4)
		{
			if (RightEdge[BOARD_WIDTH - 1 - index].stone == -1)
			{
				move = EdgeMap(LEFTEDGE, BOARD_WIDTH - 1 - index, stone);
				if (move != nullptr)
				{
					move->Set(TopEdge[index].PosX, TopEdge[index].PosY, move->TarX, move->TarY);
				}
				//////////
			}
			else if (RightEdge[BOARD_WIDTH - 1 - index].stone == stone)
			{
				if (RightEdge[BOARD_WIDTH - 1 - index].PosX == TopEdge[index].PosX && RightEdge[BOARD_WIDTH - 1 - index].PosY == TopEdge[index].PosY)
				{
					if (RightEdge[BOARD_WIDTH - 1 - index].stone2 == -1)
					{
						move = EdgeMap(LEFTEDGE, BOARD_WIDTH - 1 - index, stone);
						if (move != nullptr)
						{
							move->Set(TopEdge[index].PosX, TopEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (RightEdge[BOARD_WIDTH - 1 - index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(TopEdge[index].PosX, TopEdge[index].PosY, RightEdge[BOARD_WIDTH - 1 - index].PosX2, RightEdge[BOARD_WIDTH - 1 - index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(TopEdge[index].PosX, TopEdge[index].PosY, RightEdge[BOARD_WIDTH - 1 - index].PosX, RightEdge[BOARD_WIDTH - 1 - index].PosY);
			}
		}
	}
	else if (edgetype == BOTTOMEDGE)
	{
		if (index > 0 && index <= 2)
		{
			if (LeftEdge[BOARD_WIDTH - 1 - index].stone == -1)
			{
				move = EdgeMap(RIGHTEDGE, BOARD_WIDTH - 1 - index, stone);
				if (move != nullptr)
				{
					move->Set(BottomEdge[index].PosX, BottomEdge[index].PosY, move->TarX, move->TarY);
				}
			}
			else if (LeftEdge[BOARD_WIDTH - 1 - index].stone == stone)
			{
				if (LeftEdge[BOARD_WIDTH - 1 - index].PosX == BottomEdge[index].PosX && LeftEdge[BOARD_WIDTH - 1 - index].PosY == BottomEdge[index].PosY)
				{
					if (LeftEdge[BOARD_WIDTH - 1 - index].stone2 == -1)
					{
						move = EdgeMap(RIGHTEDGE, BOARD_WIDTH - 1 - index, stone);
						if (move != nullptr)
						{
							move->Set(BottomEdge[index].PosX, BottomEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (LeftEdge[BOARD_WIDTH - 1 - index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(BottomEdge[index].PosX, BottomEdge[index].PosY, LeftEdge[BOARD_WIDTH - 1 - index].PosX2, LeftEdge[BOARD_WIDTH - 1 - index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(BottomEdge[index].PosX, BottomEdge[index].PosY, LeftEdge[BOARD_WIDTH - 1 - index].PosX, LeftEdge[BOARD_WIDTH - 1 - index].PosY);
			}
		}
		else if (index > 2 && index <= 4)
		{
			if (RightEdge[index].stone == -1)
			{
				move = EdgeMap(LEFTEDGE, index, stone);
				if (move != nullptr)
				{
					move->Set(BottomEdge[index].PosX, BottomEdge[index].PosY, move->TarX, move->TarY);
				}
			}
			else if (RightEdge[index].stone == stone)
			{
				if (RightEdge[index].PosX == BottomEdge[index].PosX && RightEdge[index].PosY == BottomEdge[index].PosY)
				{
					if (RightEdge[index].stone2 == -1)
					{
						move = EdgeMap(LEFTEDGE, index, stone);
						if (move != nullptr)
						{
							move->Set(BottomEdge[index].PosX, BottomEdge[index].PosY, move->TarX, move->TarY);
						}
					}
					else if (RightEdge[index].stone2 == stone)
					{
						move = nullptr;
					}
					else
					{
						move = new(MOVETYPE)(BottomEdge[index].PosX, BottomEdge[index].PosY, RightEdge[index].PosX2, RightEdge[index].PosY2);
					}
				}
			}
			else
			{
				move = new(MOVETYPE)(BottomEdge[index].PosX, BottomEdge[index].PosY, RightEdge[index].PosX, RightEdge[index].PosY);
			}
		}
	}

	return move;
}



/// ��������PrintEdgemap
/// ��  ����none
/// ˵  ������ӡ����ӳ����Ϣ��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::PrintEdgemap()
{
	// ��ӡ�������Ϣ
	printf("\nLeftEdge\n");
	printf("(X,Y,stone):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", LeftEdge[i].PosX, LeftEdge[i].PosY, LeftEdge[i].stone);
	}
	printf("\n(X2,Y2,stone2):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", LeftEdge[i].PosX2, LeftEdge[i].PosY2, LeftEdge[i].stone2);
	}

	// ��ӡ�ұ�����Ϣ
	printf("\nRightEdge\n");
	printf("(X,Y,stone):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", RightEdge[i].PosX, RightEdge[i].PosY, RightEdge[i].stone);
	}
	printf("\n(X2,Y2,stone2):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", RightEdge[i].PosX2, RightEdge[i].PosY2, RightEdge[i].stone2);
	}

	// ��ӡ�ϱ�����Ϣ
	printf("\nTopEdge\n");
	printf("(X,Y,stone):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", TopEdge[i].PosX, TopEdge[i].PosY, TopEdge[i].stone);
	}
	printf("\n(X2,Y2,stone2):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", TopEdge[i].PosX2, TopEdge[i].PosY2, TopEdge[i].stone2);
	}

	// ��ӡ�±�����Ϣ
	printf("\nBottomEdge\n");
	printf("(X,Y,stone):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", BottomEdge[i].PosX, BottomEdge[i].PosY, BottomEdge[i].stone);
	}
	printf("\n(X2,Y2,stone2):");
	for (int i = 0; i < 6; i++)
	{
		printf("\t(%d,%d,%d)  ", BottomEdge[i].PosX2, BottomEdge[i].PosY2, BottomEdge[i].stone2);
	}

}

/// ��������PrintBoard
/// ��  ����none
/// ˵  ������ӡ���̡�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::PrintBoard()
{
	printf("\n");
	for (int i = 0; i < BOARD_WIDTH; i++)
	{
		printf("\n");
		for (int j = 0; j < BOARD_HEIGHT; j++)
		{
			if (Board[i][j] == 0)
			{
				printf("\t-");
			}
			else if (Board[i][j] == 1)
			{
				printf("\tx");
			}
			else if (Board[i][j] == 2)
			{
				printf("\to");
			}
		}
		printf("\n\n");
	}
	printf("\n");
}


/// ��������PrintPlayer
/// ��  ����none
/// ˵  ������ӡ��ǰ�����ӵ���ҡ�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::PrintPlayer()
{
	if (Player == HUMAN)
	{
		printf("The cunrrent move player is HUMAN. | x\n");
	}
	else if (Player == MACHINE)
	{
		printf("The cunrrent move player is MACHINE. | o\n");
	}
}

/// ��������Print
/// ��  ����none
/// ˵  ������ӡ��ǰ�����ӵ���Ҳ���ӡ���̡�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::Print()
{
	PrintPlayer();
	PrintBoard();
}

/// ��������SetPlayer
/// ��  ����PLAYER player
/// ˵  ����������ҡ�������player����Ϊ��ǰ����ҡ�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::SetPlayer(PLAYER player)
{
	Player = player;
}

/// ��������Reset
/// ��  ����none
/// ˵  ����������Ϸ���������ΪHUMAN����������ΪDefaultBoard��
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::Reset()
{
	Player = HUMAN;
	for (int i = 0; i < BOARD_WIDTH; i++)
	{
		for (int j = 0; j < BOARD_HEIGHT; j++)
		{
			Board[i][j] = DefaultBoard[i][j];
		}
	}
}

/// ��������SetBoard
/// ��  ����int board[6][6]
/// ˵  �����������̡�������board��Ϣ�����뵱ǰ�����̡�
/// ��  �ߣ��첩��
/// ʱ  �䣺2018.2.16
void GAME::SetBoard(int board[6][6])
{
	for (int i = 0; i < BOARD_WIDTH; i++)
	{
		for (int j = 0; j < BOARD_HEIGHT; j++)
		{
			Board[i][j] = board[i][j];
		}
	}
}



