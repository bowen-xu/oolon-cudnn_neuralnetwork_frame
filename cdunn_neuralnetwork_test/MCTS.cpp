/*=====================================================*
名  称：MCTS.cpp
说  明：MCTS算法模板
作  者：徐博文
版  本：V1.0.1
更  新：2018.2.17
*======================================================*/

#include "MCTS.h"
#include <windows.h>

const double Cpuct = 0.1;

NODE* NODE::getMaxSonNode()
{
	NODE* node;
	NODE* maxnode;
	node = SonNode;
	maxnode = node;

	double max;
	double buf;

	max = 0;

	while (node != nullptr)
	{
		node->U = Cpuct * node->P * sqrt(node->ParentNode->N) / (1 + node->N);
		buf = node->Q + node->U;
		if (buf > max)
		{
			maxnode = node;
			max = buf;
		}
		node = node->NextNode;
	}

	return maxnode;
}

void NODE::DelAll()
{
	if (SonNode != nullptr)
	{
		SonNode->DelAll();
	}
	if (NextNode != nullptr)
	{
		NextNode->DelAll();
	}
	delete(this);
}

NODE::~NODE()
{
	delete(Move);
}

NODE::NODE()
{
	ParentNode = nullptr;
	SonNode = nullptr;
	LastNode = nullptr;
	NextNode = nullptr;
	Move = nullptr;
}

void MCTS::Init(GAME* game)
{
	Root = new(NODE);
	Game = game;
	if (game != nullptr)
	{
		for (int i = 0; i < BOARD_HEIGHT; i++)
		{
			for (int j = 0; j < BOARD_WIDTH; j++)
			{
				Root->Board[i][j] = Game->Board[i][j];
			}
		}
	}
	
}

MCTS::MCTS(GAME* game)
{
	Init(game);
}

inline void MCTS::ExchangePlayer()
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


MOVETYPE* MCTS::getMove()
{
	NODE* leafNode;
	NODE* newNodes;

//	_LARGE_INTEGER time_start;		//开始时间
//	_LARGE_INTEGER time_over;		//结束时间
//	double dqFreq;					//计时器频率
//	LARGE_INTEGER f;				//计时器频率
//#define delta_time ((time_over.QuadPart - time_start.QuadPart) / dqFreq * 1000)
//	QueryPerformanceFrequency(&f);
//	dqFreq = (double)f.QuadPart;
//
//	getchar();
//	QueryPerformanceCounter(&time_start);
	
	for (int i = 0; i < 1400; i++)
	{
	leafNode = Select();
	Expand(leafNode);
	//if (newNodes == nullptr)
	//{
	//	continue;
	//}
	Evaluate(leafNode);
	Backup(leafNode);
	}

	//QueryPerformanceCounter(&time_over);
	//printf("CPU运算时间：%.5f ms\n", delta_time);
	//printf("平均运算时间：%.5f ms\n", delta_time/1400);


	return nullptr;
}

inline void MCTS::CopyBoard(int srcBoard[BOARD_HEIGHT][BOARD_WIDTH], int tarBoard[BOARD_HEIGHT][BOARD_WIDTH])
{
	for (int i = 0; i < BOARD_HEIGHT; i++)
	{
		for (int j = 0; j < BOARD_WIDTH; j++)
		{
			tarBoard[i][j] = srcBoard[i][j];
		}
	}
}

// 步骤1：选择结点
NODE* MCTS::Select()
{
	int i = 0;
	NODE* node;
	node = Root;
	Player = Game->Player;
	// 叶子结点退出循环
	while (node->SonNode != nullptr)
	{
		node = node->getMaxSonNode();
		i++;
	}

	if (i % 2 == 1)
	{
		ExchangePlayer();
	}

	return node;
}

// 步骤2：扩展结点
void MCTS::Expand(NODE* leafnode)
{
	MOVELIST* moves;
	NODE* newNode;
	
	moves = Game->GetMoves(Player, leafnode->Board);
	//moves->Print();
	while (moves != nullptr)
	{
		newNode = new(NODE);
		newNode->Move = moves->Move;
		moves->Move = nullptr;
		moves->Del(&moves);

		newNode->ParentNode = leafnode;

		CopyBoard(leafnode->Board, newNode->Board);
		Game->Move(*(newNode->Move), newNode->Board);

		// 链表操作
		if (leafnode->SonNode == nullptr)
		{
			leafnode->SonNode = newNode;
		}
		else
		{
			if (leafnode->SonNode->NextNode != nullptr)
			{
				leafnode->SonNode->NextNode->LastNode = newNode;
			}
			newNode->NextNode = leafnode->SonNode->NextNode;
			leafnode->SonNode->NextNode = newNode;
			newNode->LastNode = leafnode->SonNode;
		}
	}

	//return leafnode->SonNode;
}

// 步骤3：评估结点
void MCTS::Evaluate(NODE* node)
{
#define ResNet 1
	node->V = ResNet;
	node = node->SonNode;
	while (node != nullptr)
	{
		// 神经网路估计结果
		node->P = ResNet;
		node = node->NextNode;
	}
}

// 步骤4：反向传播
void MCTS::Backup(NODE* node)
{
	double value;

	value = node->V;

	while (node != nullptr)
	{
		node->N += 1;
		node->W += value;
		node->Q = node->W / node->N;
		node = node->ParentNode;
	}
}