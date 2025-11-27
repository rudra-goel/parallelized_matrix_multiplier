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
int windowWidth = 800;

/** @brief Height of the SFML render window. */
int windowHeight = 600;

int matrixWidth = 32;
int matrixHeight = 32;

int mat[32][32];

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
            mat[i][j] = distrib(gen);
        }
    }

    // Create render window
    sf::VideoMode vm(windowWidth, windowHeight);
    sf::RenderWindow window(vm, "CUDA Matrix Multiplication Visualization", sf::Style::Default);

    sf::Clock clock;
    sf::Clock processingClock;
    long long processingTime = 0;

    sf::RectangleShape matrixOutline(sf::Vector2f(500, 500));
    matrixOutline.setOutlineColor(sf::Color::Black);
    matrixOutline.setPosition(150, 50);
    matrixOutline.setOutlineThickness(1);
    sf::VertexArray matrixLines(sf::Lines, (matrixWidth - 1) * 2 + (matrixHeight - 1) * 2);
    for (int i = 1; i <= matrixLines.getVertexCount(); i++)
    {
        matrixLines[i-1].color = sf::Color::Black;
        // Vertical lines
        if (i <= matrixLines.getVertexCount() / 2)
        {
            if (i % 2 == 1)
            {
                matrixLines[i-1].position = sf::Vector2f(150 + 500.0f / matrixWidth * std::ceil(i / 2.0f), 50);
            }
            else
            {
                matrixLines[i-1].position = sf::Vector2f(150 + 500.0f / matrixWidth * (i / 2), 550);
            }
        }
        // Horizontal lines
        else
        {
            int j = i - matrixLines.getVertexCount() / 2;
            if (j % 2 == 1)
            {
                matrixLines[i-1].position = sf::Vector2f(150, 50 + 500.0f / matrixHeight * std::ceil(j / 2.0f));
            }
            else
            {
                matrixLines[i-1].position = sf::Vector2f(650, 50 + 500.0f / matrixHeight * (j / 2));
            }
        }
    }

    sf::Font font;
    if (!font.loadFromFile("arial.ttf"))
    {
        std::cerr << "Arial font was not loaded! Exiting" << std::endl;
        return 1;
    }
    sf::Text numberTexts[32 * 32];
    for (int i = 0; i < 32 * 32; i++)
    {
        numberTexts[i].setFont(font);
        numberTexts[i].setString(std::to_string(mat[i / 32][i % 32]));
        numberTexts[i].setCharacterSize(16);
        numberTexts[i].setFillColor(sf::Color::Red);
        numberTexts[i].setPosition(153.5 + 500.0f / 32 * (i % 32), 48 + 500.0f / 32 * (i / 32));
    }

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

        window.clear(sf::Color::White);
        window.draw(matrixOutline);
        window.draw(matrixLines);
        for (int i = 0; i < 32 * 32; i++)
        {
            window.draw(numberTexts[i]);
        }
        window.display();
        std::cout << "FPS: " << 1.0f / processingClock.restart().asSeconds() << std::endl;
        
        
        // Measure processing time
        processingTime += processingClock.getElapsedTime().asMicroseconds();
    }

    return EXIT_SUCCESS;
}