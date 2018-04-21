/*=====================================================*
��  �ƣ�MCTS.h
˵  ����MCTS�㷨ģ��
��  �ߣ��첩��
��  ����V1.0.1
��  �£�2018.2.17
*======================================================*/

#ifndef _MCTS_H_
#define _MCTS_H_
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "Surakarta.h"



// �����
class NODE
{
public:
	friend class MCTS;
	NODE* getMaxSonNode();
	void DelAll();
	NODE();
	~NODE();
private:
	NODE* ParentNode;
	NODE* SonNode;
	NODE* LastNode;
	NODE* NextNode;
	MOVETYPE* Move;
	int Board[BOARD_HEIGHT][BOARD_WIDTH];
	int		N = 0;	// visit count
	int		W = 0;	// total action value
	double	Q = 0;	// mean action value
	double	P = 0;	// prior propability of selection
	double	U = 0;	// 
	double	V = 0;	// value evaluated by neural network
};

// MCTS��
class MCTS
{
public:
	friend class NODE;
	void Init(GAME* game);
	MCTS(GAME* game = nullptr);
	MOVETYPE* getMove();
private:
	NODE* Root;
	GAME* Game;
	PLAYER Player;
	inline void ExchangePlayer();
	inline void CopyBoard(int srcBoard[BOARD_HEIGHT][BOARD_WIDTH], int tarBoard[BOARD_HEIGHT][BOARD_WIDTH]);
	NODE* Select();
	void Expand(NODE* node);
	void Evaluate(NODE* node);
	void Backup(NODE* node);
};

#endif // !_MCTS_H_
