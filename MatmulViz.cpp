/* 
Author: Jackie Mac Hale
Class: ECE6122 A
Last Date Modified: 10/3/2025

Description: 

This file contains all of the window rendering and pixel drawing for Conway's
Game Of Life. It supports single-threaded, multi-threaded (std::thread), and
OpenMP-based parallelism.

*/ 

#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>

/** @brief Width of the SFML render window. */
int windowWidth = 1920;

/** @brief Height of the SFML render window. */
int windowHeight = 1080;

int matrixWidth = 32;
int matrixHeight = 32;

int matA[32][32];
int matB[32][32];
int matOut[32][32];

int mainMatrix(int argc, char **argv);

/**
 * @brief Entry point for the Game of Life application.
 * 
 * Supports command-line arguments for cell size, window dimensions,
 * number of threads, and threading mode.
 * 
 * @param argc Argument count
 * @param argv Argument vector
 * @return int EXIT_SUCCESS if the game exits normally, EXIT_FAILURE otherwise
 */

int main(int argc, char* argv[])
{
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 gen(seed);

    std::uniform_int_distribution<int> distrib(0, 9);
    // mainMatrix(argc, argv);

    for (int i = 0; i < matrixHeight; i++)
    {
        for (int j = 0; j < matrixWidth; j++)
        {
            matA[i][j] = distrib(gen);
            matB[i][j] = distrib(gen);
        }
    }

    // for (int i = 0; i < matrixHeight; i++)
    // {
    //     for (int j = 0; j < matrixWidth; j++)
    //     {
    //         matOut[i][j] = 0;
    //         for (int k = 0; k < matrixWidth; k++)
    //         {
    //             matOut[i][j] += matA[i][k] * matB[k][j];
    //         }
    //     }
    // }

    // Create render window
    sf::VideoMode vm(windowWidth, windowHeight);
    sf::RenderWindow window(vm, "CUDA Matrix Multiplication Visualization", sf::Style::Default);

    sf::Clock clock;
    sf::Clock processingClock;
    long long processingTime = 0;

    int matALeftEdgeX = 25;
    int matATopEdgeY = 200;
    float matAHeight = 400.0f;
    int matBLeftEdgeX = 462.5;
    int matBTopEdgeY = 200;
    float matBHeight = matAHeight;
    int matOutLeftEdgeX = 900;
    float matOutHeight = 800.0f;

    sf::RectangleShape matAOutline(sf::Vector2f(matAHeight, matAHeight));
    sf::RectangleShape matBOutline(sf::Vector2f(matBHeight, matBHeight));
    sf::RectangleShape matOutOutline(sf::Vector2f(matOutHeight, matOutHeight));
    matAOutline.setOutlineColor(sf::Color::Black);
    matBOutline.setOutlineColor(sf::Color::Black);
    matOutOutline.setOutlineColor(sf::Color::Black);
    matAOutline.setPosition(matALeftEdgeX, matATopEdgeY);
    matBOutline.setPosition(matBLeftEdgeX, matBTopEdgeY);
    matOutOutline.setPosition(matOutLeftEdgeX, 50);
    matAOutline.setOutlineThickness(1);
    matBOutline.setOutlineThickness(1);
    matOutOutline.setOutlineThickness(1);
    sf::VertexArray matALines(sf::Lines, (matrixWidth - 1) * 2 + (matrixHeight - 1) * 2);
    sf::VertexArray matBLines(sf::Lines, (matrixWidth - 1) * 2 + (matrixHeight - 1) * 2);
    sf::VertexArray matOutLines(sf::Lines, (matrixWidth - 1) * 2 + (matrixHeight - 1) * 2);
    for (int i = 1; i <= matALines.getVertexCount(); i++)
    {
        matALines[i-1].color = sf::Color::Black;
        matBLines[i-1].color = sf::Color::Black;
        matOutLines[i-1].color = sf::Color::Black;
        // Vertical lines
        if (i <= matALines.getVertexCount() / 2)
        {
            if (i % 2 == 1)
            {
                matALines[i-1].position = sf::Vector2f(matALeftEdgeX + matAHeight / matrixWidth * std::ceil(i / 2.0f), matATopEdgeY);
                matBLines[i-1].position = sf::Vector2f(matBLeftEdgeX + matBHeight / matrixWidth * std::ceil(i / 2.0f), matBTopEdgeY);
                matOutLines[i-1].position = sf::Vector2f(matOutLeftEdgeX + matOutHeight / matrixWidth * std::ceil(i / 2.0f), 50);
            }
            else
            {
                matALines[i-1].position = sf::Vector2f(matALeftEdgeX + matAHeight / matrixWidth * (i / 2), matAHeight + matATopEdgeY);
                matBLines[i-1].position = sf::Vector2f(matBLeftEdgeX + matBHeight / matrixWidth * (i / 2), matBHeight + matBTopEdgeY);
                matOutLines[i-1].position = sf::Vector2f(matOutLeftEdgeX + matOutHeight / matrixWidth * (i / 2), matOutHeight + 50);
            }
        }
        // Horizontal lines
        else
        {
            int j = i - matALines.getVertexCount() / 2;
            if (j % 2 == 1)
            {
                matALines[i-1].position = sf::Vector2f(matALeftEdgeX, matATopEdgeY + matAHeight / matrixHeight * std::ceil(j / 2.0f));
                matBLines[i-1].position = sf::Vector2f(matBLeftEdgeX, matBTopEdgeY + matBHeight / matrixHeight * std::ceil(j / 2.0f));
                matOutLines[i-1].position = sf::Vector2f(matOutLeftEdgeX, 50 + matOutHeight / matrixHeight * std::ceil(j / 2.0f));
            }
            else
            {
                matALines[i-1].position = sf::Vector2f(matALeftEdgeX + matAHeight, matATopEdgeY + matAHeight / matrixHeight * (j / 2));
                matBLines[i-1].position = sf::Vector2f(matBLeftEdgeX + matBHeight, matBTopEdgeY + matBHeight / matrixHeight * (j / 2));
                matOutLines[i-1].position = sf::Vector2f(matOutLeftEdgeX + matOutHeight, 50 + matOutHeight / matrixHeight * (j / 2));
            }
        }
    }

    sf::Font font;
    if (!font.loadFromFile("arial.ttf"))
    {
        std::cerr << "Arial font was not loaded! Exiting" << std::endl;
        return 1;
    }
    sf::Text matANumberTexts[32 * 32];
    sf::Text matBNumberTexts[32 * 32];
    sf::Text matOutNumberTexts[32 * 32];
    for (int i = 0; i < 32 * 32; i++)
    {
        matANumberTexts[i].setFont(font);
        matANumberTexts[i].setString(std::to_string(matA[i / 32][i % 32]));
        matANumberTexts[i].setCharacterSize(12);
        matANumberTexts[i].setFillColor(sf::Color::Red);
        matANumberTexts[i].setPosition(matALeftEdgeX + 2.5 + matAHeight / 32 * (i % 32), matATopEdgeY - 1.5 + matAHeight / 32 * (i / 32));

        matBNumberTexts[i].setFont(font);
        matBNumberTexts[i].setString(std::to_string(matB[i / 32][i % 32]));
        matBNumberTexts[i].setCharacterSize(12);
        matBNumberTexts[i].setFillColor(sf::Color::Red);
        matBNumberTexts[i].setPosition(matBLeftEdgeX + 2.5 + matBHeight / 32 * (i % 32), matBTopEdgeY - 1.5 + matBHeight / 32 * (i / 32));

        matOutNumberTexts[i].setFont(font);
        matOutNumberTexts[i].setString(std::to_string(matOut[i / 32][i % 32]));
        matOutNumberTexts[i].setCharacterSize(10);
        matOutNumberTexts[i].setFillColor(sf::Color::Blue);
        matOutNumberTexts[i].setPosition(matOutLeftEdgeX + 9.5 + matOutHeight / 32 * (i % 32), 56 + matOutHeight / 32 * (i / 32));
    }

    sf::Text multiplySign;
    sf::Text equalSign;
    multiplySign.setFont(font);
    equalSign.setFont(font);
    multiplySign.setString("x");
    equalSign.setString("=");
    multiplySign.setCharacterSize(18);
    equalSign.setCharacterSize(18);
    multiplySign.setFillColor(sf::Color::Black);
    equalSign.setFillColor(sf::Color::Black);
    multiplySign.setPosition(matBLeftEdgeX - 24, matATopEdgeY + matAHeight / 2 - 25);
    equalSign.setPosition(matOutLeftEdgeX - 24, matATopEdgeY + matAHeight / 2 - 23);

    sf::RectangleShape matACurrElem(sf::Vector2f(matAHeight / 32.0f, matAHeight / 32.0f));
    sf::RectangleShape matBCurrElem(sf::Vector2f(matBHeight / 32.0f, matBHeight / 32.0f));
    sf::RectangleShape matOutCurrElem(sf::Vector2f(matOutHeight / 32.0f, matOutHeight / 32.0f));
    matACurrElem.setPosition(matALeftEdgeX, matATopEdgeY);
    matBCurrElem.setPosition(matBLeftEdgeX, matBTopEdgeY);
    matOutCurrElem.setPosition(matOutLeftEdgeX, 50);
    matACurrElem.setFillColor(sf::Color(51, 153, 255));
    matBCurrElem.setFillColor(sf::Color(51, 153, 255));
    matOutCurrElem.setFillColor(sf::Color::Red);

    int iteration = 0;
    sf::Clock iterationTimer;
    iterationTimer.restart();

    float matACurrElemX = matALeftEdgeX;
    float matACurrElemY = matATopEdgeY;
    float matBCurrElemX = matBLeftEdgeX;
    float matBCurrElemY = matBTopEdgeY;
    float matOutCurrElemX = matOutLeftEdgeX;
    float matOutCurrElemY = 50;

    matOut[0][0] = matA[0][0] * matB[0][0];
    matOutNumberTexts[0].setString(std::to_string(matOut[0][0]));
    std::string str = matOutNumberTexts[0].getString();
    if (str.length() == 2)
    {
        matOutNumberTexts[0].setPosition(matOutLeftEdgeX + 6.5, matOutNumberTexts[0].getPosition().y);
    }

    int i = 0;
    int j = 0;
    int k = 0;

    // Main Game Loop
    while (window.isOpen())
    {
        // Quit game if ESC is pressed
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
		{
			window.close();
		}

        // Poll window events
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
        }

        if (iterationTimer.getElapsedTime().asMilliseconds() > 20)
        {
            iterationTimer.restart();
            if (iteration < 32 * 32 * 32 - 1)
            {
                ++iteration;
                if (iteration % (32 * 32) == 0)
                {
                    i++;
                    j = 0;
                    k = 0;
                    matACurrElemX = matALeftEdgeX;
                    matACurrElemY += matAHeight / 32.0f;
                    matBCurrElemX = matBLeftEdgeX;
                    matBCurrElemY = matBTopEdgeY;
                    matOutCurrElemX = matOutLeftEdgeX;
                    matOutCurrElemY += matOutHeight / 32;
                }
                else if (iteration % 32 == 0)
                {
                    j++;
                    k = 0;
                    matACurrElemX = matALeftEdgeX;
                    matBCurrElemX += matBHeight / 32;
                    matBCurrElemY = matBTopEdgeY;
                    matOutCurrElemX += matOutHeight / 32;
                }
                else
                {
                    k++;
                    matACurrElemX += matAHeight / 32;
                    matBCurrElemY += matBHeight / 32;
                }
                matOut[i][j] += matA[i][k] * matB[k][j];
                matOutNumberTexts[32 * i + j].setString(std::to_string(matOut[i][j]));
                str = std::to_string(matOut[i][j]);
                if (str.length() == 2)
                {
                    matOutNumberTexts[32 * i + j].setPosition(matOutLeftEdgeX + 6.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[32 * i + j].getPosition().y);
                }
                else if (str.length() == 3)
                {
                    matOutNumberTexts[32 * i + j].setPosition(matOutLeftEdgeX + 3.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[32 * i + j].getPosition().y);
                }
                else if (str.length() == 4)
                {
                    matOutNumberTexts[32 * i + j].setPosition(matOutLeftEdgeX + 0.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[32 * i + j].getPosition().y);
                }
            }
        }

        matACurrElem.setPosition(matACurrElemX, matACurrElemY);
        matBCurrElem.setPosition(matBCurrElemX, matBCurrElemY);
        matOutCurrElem.setPosition(matOutCurrElemX, matOutCurrElemY);

        window.clear(sf::Color::White);
        window.draw(matAOutline);
        window.draw(matBOutline);
        window.draw(matOutOutline);
        window.draw(matACurrElem);
        window.draw(matBCurrElem);
        window.draw(matOutCurrElem);
        window.draw(matALines);
        window.draw(matBLines);
        window.draw(matOutLines);
        window.draw(multiplySign);
        window.draw(equalSign);
        for (int i = 0; i < 32 * 32; i++)
        {
            window.draw(matANumberTexts[i]);
            window.draw(matBNumberTexts[i]);
            window.draw(matOutNumberTexts[i]);
        }
        window.display();
        std::cout << "FPS: " << 1.0f / processingClock.restart().asSeconds() << std::endl;
        
        
        // Measure processing time
        processingTime += processingClock.getElapsedTime().asMicroseconds();
    }

    return EXIT_SUCCESS;
}