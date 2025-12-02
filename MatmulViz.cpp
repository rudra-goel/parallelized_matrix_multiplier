/* 
Author: Rudra Goel, Jackie Mac Hale
Class: ECE4122 A, ECE6122 A
Last Date Modified: 12/2/2025

Description: 

This program visualizes matrix multiplication using three different techniques.
The first is using a CPU to perform matrix multiplication, the second is
naively using a GPU with CUDA to parallelize the matrix multiplication
operations without reusing matrix values, and the third option is optimizing
memory accesses using a GPU with CUDA by tiling the two input matrices using
shared buffers that decrease the amount of global memory accesses, speeding up
the program. The program also allows an input file to be specified in CSV format
for the two input matrices used in the visualization.

*/

#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>

/** @brief Width of the SFML render window. */
int windowWidth = 1920;

/** @brief Height of the SFML render window. */
int windowHeight = 1080;

/** @brief Width of the square matrices. */
int matrixWidth = 32;

/** @brief Height of the square matrices. */
int matrixHeight = 32;

/** @brief Matrix A for multiplication (32x32). */
int matA[32][32];

/** @brief Matrix B for multiplication (32x32). */
int matB[32][32];

/** @brief Output matrix for multiplication (32x32). */
int matOut[32][32];

/** @brief Controls the matrix multiplication method used.
 * 0: CPU-like (element-wise dot product for result cell)
 * 1: Naive CUDA (row-column dot product for each cell)
 * 2: Efficient Tiling (block-based multiplication)
 */
int mode = 0;

/**
 * @brief Placeholder function for performing the actual matrix multiplication based on the selected mode.
 *
 * @param modeArg The selected mode for matrix multiplication (0: CPU, 1: Naive CUDA, 2: Tiling)
 * @return int The status of the matrix multiplication
 */
int mainMatrix(int modeArg);
int mainMatrix(int modeArg);

/**
 * @brief Entry point for the CUDA Matrix Multiplication Visualization application.
 * 
 * Supports command-line arguments to specify the multiplication method and input data file.
 * The core functionality sets up the matrix data, performs an initial matrix calculation, 
 * and then enters an SFML render loop to visualize the multiplication process step-by-step.
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return int 0 if the application exits normally, 1 otherwise
 */

int main(int argc, char* argv[])
{
    // Initialize a random number generator using the current system time as the seed
    unsigned seed = static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count());
    std::mt19937 gen(seed);

    // Distribution for generating random matrix elements between 0 and 9
    std::uniform_int_distribution<int> distrib(0, 9);

    // Command-line argument strings
    std::string modeArg = "--method";
    std::string naiveMethod = "n";
    std::string tilingMethod = "ef";
    std::string cpuMethod = "c";
    std::string fileArg = "--data";

    bool dataFileProvided = false;

    // Parses command-line arguments. Iterates over arguments in pairs (flag and value)
    for (int i = 1; i < argc - 1; i += 2)
    {
        // Check for the method argument
        if (argv[i] == modeArg)
        {
            if (argv[i+1] == naiveMethod)
            {
                mode = 1;
            }
            else if (argv[i+1] == tilingMethod)
            {
                mode = 2;
            }
            else if (argv[i+1] == cpuMethod)
            {
                mode = 0;
            }
            else
            {
                std::cerr << "Unrecognized option for performing matrix multiplication. Recognized options are n for naive, ef for efficient tiling, and c for cpu methods." << std::endl;
                return 1;
            }
        }
        // Check for the data file argument
        else if (argv[i] == fileArg)
        {
            std::ifstream inFile(argv[i+1]);
            if (!inFile.is_open())
            {
                std::cerr << "Error opening file for reading. Check that the file exists." << std::endl;
                return 1;
            }

            std::string line;
            int i = 0;
            // Read file line by line
            while (std::getline(inFile, line))
            {
                int j = 0;
                std::stringstream ss(line);
                std::string cell;
                // Parse a line of comma-separated values (CSV)
                while (std::getline(ss, cell, ','))
                {
                    if (i < 32)
                    {
                        matA[i][j] = std::stoi(cell);
                    }
                    else
                    {
                        matB[i % 32][j] = std::stoi(cell);
                    }
                    j++;
                }
                i++;
            }

            dataFileProvided = true;
        }
    }

    // If no data file was provided, initialize matrices A and B with random values
    if (!dataFileProvided)
    {
        for (int i = 0; i < matrixHeight; i++)
        {
            for (int j = 0; j < matrixWidth; j++)
            {
                matA[i][j] = distrib(gen);
                matB[i][j] = distrib(gen);
            }
        }
    }

    // Call the CUDA matrix multiplication code
    mainMatrix(mode);

    // --- SFML Setup and Initialization ---

    // Create render window
    sf::VideoMode vm(windowWidth, windowHeight);
    sf::RenderWindow window(vm, "CUDA Matrix Multiplication Visualization", sf::Style::Default);

    // Clock for frame timing
    sf::Clock clock;

    // Define coordinates and dimensions for the matrices in the visualization
    int matALeftEdgeX = 25;
    int matATopEdgeY = 200;
    float matAHeight = 400.0f;
    int matBLeftEdgeX = 462.5;
    int matBTopEdgeY = 200;
    float matBHeight = matAHeight;
    int matOutLeftEdgeX = 900;
    float matOutHeight = 800.0f;

    // Create outlines for the 3 matrices (A, B, and output)
    sf::RectangleShape matAOutline(sf::Vector2f(matAHeight, matAHeight));
    sf::RectangleShape matBOutline(sf::Vector2f(matBHeight, matBHeight));
    sf::RectangleShape matOutOutline(sf::Vector2f(matOutHeight, matOutHeight));

    // Set outline appearance and position
    matAOutline.setOutlineColor(sf::Color::Black);
    matBOutline.setOutlineColor(sf::Color::Black);
    matOutOutline.setOutlineColor(sf::Color::Black);
    matAOutline.setPosition(matALeftEdgeX, matATopEdgeY);
    matBOutline.setPosition(matBLeftEdgeX, matBTopEdgeY);
    matOutOutline.setPosition(matOutLeftEdgeX, 50);
    matAOutline.setOutlineThickness(1);
    matBOutline.setOutlineThickness(1);
    matOutOutline.setOutlineThickness(1);

    // Create VertexArrays to draw grid lines for the matrices
    sf::VertexArray matALines(sf::Lines, (matrixWidth - 1) * 2 + (matrixHeight - 1) * 2);
    sf::VertexArray matBLines(sf::Lines, (matrixWidth - 1) * 2 + (matrixHeight - 1) * 2);
    sf::VertexArray matOutLines(sf::Lines, (matrixWidth - 1) * 2 + (matrixHeight - 1) * 2);

    // Calculate and set the positions for all grid lines (vertical and horizontal)
    for (int i = 1; i <= matALines.getVertexCount(); i++)
    {
        matALines[i-1].color = sf::Color::Black;
        matBLines[i-1].color = sf::Color::Black;
        matOutLines[i-1].color = sf::Color::Black;

        // Vertical lines
        if (i <= matALines.getVertexCount() / 2)
        {
            // Start point (top)
            if (i % 2 == 1)
            {
                matALines[i-1].position = sf::Vector2f(matALeftEdgeX + matAHeight / matrixWidth * std::ceil(i / 2.0f), matATopEdgeY);
                matBLines[i-1].position = sf::Vector2f(matBLeftEdgeX + matBHeight / matrixWidth * std::ceil(i / 2.0f), matBTopEdgeY);
                matOutLines[i-1].position = sf::Vector2f(matOutLeftEdgeX + matOutHeight / matrixWidth * std::ceil(i / 2.0f), 50);
            }
            // End point (bottom)
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
            // Start point (left)
            if (j % 2 == 1)
            {
                matALines[i-1].position = sf::Vector2f(matALeftEdgeX, matATopEdgeY + matAHeight / matrixHeight * std::ceil(j / 2.0f));
                matBLines[i-1].position = sf::Vector2f(matBLeftEdgeX, matBTopEdgeY + matBHeight / matrixHeight * std::ceil(j / 2.0f));
                matOutLines[i-1].position = sf::Vector2f(matOutLeftEdgeX, 50 + matOutHeight / matrixHeight * std::ceil(j / 2.0f));
            }
            // End point (right)
            else
            {
                matALines[i-1].position = sf::Vector2f(matALeftEdgeX + matAHeight, matATopEdgeY + matAHeight / matrixHeight * (j / 2));
                matBLines[i-1].position = sf::Vector2f(matBLeftEdgeX + matBHeight, matBTopEdgeY + matBHeight / matrixHeight * (j / 2));
                matOutLines[i-1].position = sf::Vector2f(matOutLeftEdgeX + matOutHeight, 50 + matOutHeight / matrixHeight * (j / 2));
            }
        }
    }

    // Load the font for displaying matrix element values
    sf::Font font;
    if (!font.loadFromFile("arial.ttf"))
    {
        std::cerr << "Arial font was not loaded! Exiting" << std::endl;
        return 1;
    }

    // Create SFML Text objects for displaying all matrix elements
    sf::Text matANumberTexts[32 * 32];
    sf::Text matBNumberTexts[32 * 32];
    sf::Text matOutNumberTexts[32 * 32];

    // Initialize the text properties and positions for all elements in the matrices
    for (int i = 0; i < 32 * 32; i++)
    {
        // Matrix A text
        matANumberTexts[i].setFont(font);
        matANumberTexts[i].setString(std::to_string(matA[i / 32][i % 32]));
        matANumberTexts[i].setCharacterSize(12);
        matANumberTexts[i].setFillColor(sf::Color::Red);
        matANumberTexts[i].setPosition(matALeftEdgeX + 2.5 + matAHeight / 32 * (i % 32), matATopEdgeY - 1.5 + matAHeight / 32 * (i / 32));

        // Matrix B text
        matBNumberTexts[i].setFont(font);
        matBNumberTexts[i].setString(std::to_string(matB[i / 32][i % 32]));
        matBNumberTexts[i].setCharacterSize(12);
        matBNumberTexts[i].setFillColor(sf::Color::Red);
        matBNumberTexts[i].setPosition(matBLeftEdgeX + 2.5 + matBHeight / 32 * (i % 32), matBTopEdgeY - 1.5 + matBHeight / 32 * (i / 32));

        // Output matrix text
        matOutNumberTexts[i].setFont(font);
        matOutNumberTexts[i].setString(std::to_string(matOut[i / 32][i % 32]));
        matOutNumberTexts[i].setCharacterSize(10);
        matOutNumberTexts[i].setFillColor(sf::Color::Blue);
        matOutNumberTexts[i].setPosition(matOutLeftEdgeX + 9.5 + matOutHeight / 32 * (i % 32), 56 + matOutHeight / 32 * (i / 32));
    }

    // Create and position the multiplication and equals signs
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

    // --- Tiling Mode Visualization Elements ---

    // Rectangles to highlight the current tiles being processed in matrices A, B, and output
    sf::RectangleShape matATileRect(sf::Vector2f(matAHeight / 8.0f, matAHeight / 8.0f));
    sf::RectangleShape matBTileRect(sf::Vector2f(matBHeight / 8.0f, matBHeight / 8.0f));
    sf::RectangleShape matOutTileRect(sf::Vector2f(matOutHeight / 8.0f, matOutHeight / 8.0f));

    // Rectangles to represent shared memory for visualization
    sf::RectangleShape matASharedMem(sf::Vector2f(matAHeight / 8.0f, matAHeight / 8.0f));
    sf::RectangleShape matBSharedMem(sf::Vector2f(matBHeight / 8.0f, matBHeight / 8.0f));

    // VertexArrays for grid lines within the shared memory representation
    sf::VertexArray matASharedMemLines(sf::Lines, (matrixWidth / 8 - 1) * 2 + (matrixHeight / 8 - 1) * 2);
    sf::VertexArray matBSharedMemLines(sf::Lines, (matrixWidth / 8 - 1) * 2 + (matrixHeight / 8 - 1) * 2);

    // Positions and dimensions for the shared memory grids
    float matASharedMemLeftEdgeX = 200;
    float matASharedMemTopEdgeY = 700;
    float matASharedMemHeight = matAHeight / 8;
    float matBSharedMemLeftEdgeX = 637.5;
    float matBSharedMemTopEdgeY = matASharedMemTopEdgeY;
    float matBSharedMemHeight = matASharedMemHeight;

    // Set appearance and position for shared memory grid outlines
    matASharedMem.setOutlineColor(sf::Color(255, 165, 0));
    matASharedMem.setPosition(matASharedMemLeftEdgeX, matASharedMemTopEdgeY);
    matASharedMem.setOutlineThickness(1);
    matBSharedMem.setOutlineColor(sf::Color(255, 165, 0));
    matBSharedMem.setPosition(matBSharedMemLeftEdgeX, matBSharedMemTopEdgeY);
    matBSharedMem.setOutlineThickness(1);

    // Set the positions for grid lines within the shared memory grids
    for (int i = 1; i <= matASharedMemLines.getVertexCount(); i++)
    {
        matASharedMemLines[i-1].color = sf::Color::Black;
        matBSharedMemLines[i-1].color = sf::Color::Black;

        // Vertical lines
        if (i <= matASharedMemLines.getVertexCount() / 2)
        {
            // Start point (top)
            if (i % 2 == 1)
            {
                matASharedMemLines[i-1].position = sf::Vector2f(matASharedMemLeftEdgeX + matASharedMemHeight / (matrixWidth / 8) * std::ceil(i / 2.0f), matASharedMemTopEdgeY);
                matBSharedMemLines[i-1].position = sf::Vector2f(matBSharedMemLeftEdgeX + matBSharedMemHeight / (matrixWidth / 8) * std::ceil(i / 2.0f), matBSharedMemTopEdgeY);
            }
            // End point (bottom)
            else
            {
                matASharedMemLines[i-1].position = sf::Vector2f(matASharedMemLeftEdgeX + matASharedMemHeight / (matrixWidth / 8) * (i / 2), matASharedMemHeight + matASharedMemTopEdgeY);
                matBSharedMemLines[i-1].position = sf::Vector2f(matBSharedMemLeftEdgeX + matBSharedMemHeight / (matrixWidth / 8) * (i / 2), matBSharedMemHeight + matBSharedMemTopEdgeY);
            }
        }
        // Horizontal lines
        else
        {
            int j = i - matASharedMemLines.getVertexCount() / 2;
            // Start point (left)
            if (j % 2 == 1)
            {
                matASharedMemLines[i-1].position = sf::Vector2f(matASharedMemLeftEdgeX, matASharedMemTopEdgeY + matASharedMemHeight / (matrixHeight / 8) * std::ceil(j / 2.0f));
                matBSharedMemLines[i-1].position = sf::Vector2f(matBSharedMemLeftEdgeX, matBSharedMemTopEdgeY + matBSharedMemHeight / (matrixHeight / 8) * std::ceil(j / 2.0f));
            }
            // End point (right)
            else
            {
                matASharedMemLines[i-1].position = sf::Vector2f(matASharedMemLeftEdgeX + matASharedMemHeight, matASharedMemTopEdgeY + matASharedMemHeight / (matrixHeight / 8) * (j / 2));
                matBSharedMemLines[i-1].position = sf::Vector2f(matBSharedMemLeftEdgeX + matBSharedMemHeight, matBSharedMemTopEdgeY + matBSharedMemHeight / (matrixHeight / 8) * (j / 2));
            }
        }
    }

    // Text objects for displaying elements in the shared memory grids
    sf::Text matASharedMemNumberTexts[4 * 4];
    sf::Text matBSharedMemNumberTexts[4 * 4];

    // Initialize the text for the shared memory grids
    for (int i = 0; i < 4 * 4; i++)
    {
        // Matrix A shared memory text
        matASharedMemNumberTexts[i].setFont(font);
        matASharedMemNumberTexts[i].setString(std::to_string(matA[i / 4][i % 4]));
        matASharedMemNumberTexts[i].setCharacterSize(12);
        matASharedMemNumberTexts[i].setFillColor(sf::Color::Red);
        matASharedMemNumberTexts[i].setPosition(matASharedMemLeftEdgeX + 2.5 + matASharedMemHeight / 4 * (i % 4), matASharedMemTopEdgeY - 1.5 + matASharedMemHeight / 4 * (i / 4));

        // Matrix B shared memory text
        matBSharedMemNumberTexts[i].setFont(font);
        matBSharedMemNumberTexts[i].setString(std::to_string(matB[i / 4][i % 4]));
        matBSharedMemNumberTexts[i].setCharacterSize(12);
        matBSharedMemNumberTexts[i].setFillColor(sf::Color::Red);
        matBSharedMemNumberTexts[i].setPosition(matBSharedMemLeftEdgeX + 2.5 + matBSharedMemHeight / 4 * (i % 4), matBSharedMemTopEdgeY - 1.5 + matBSharedMemHeight / 4 * (i / 4));
    }

    // Naive CUDA Visualization Elements
    sf::RectangleShape matACurrRowRect(sf::Vector2f(matAHeight, matAHeight / 32.0f));
    sf::RectangleShape matBCurrColRect(sf::Vector2f(matBHeight / 32.0f, matBHeight));

    // CPU Visualization Elements
    sf::RectangleShape matACurrElem(sf::Vector2f(matAHeight / 32.0f, matAHeight / 32.0f));
    sf::RectangleShape matBCurrElem(sf::Vector2f(matBHeight / 32.0f, matBHeight / 32.0f));
    sf::RectangleShape matOutCurrElem(sf::Vector2f(matOutHeight / 32.0f, matOutHeight / 32.0f));

    // Set initial positions and colors for current element/row/column for naive CUDA and CPU modes
    matACurrElem.setPosition(matALeftEdgeX, matATopEdgeY);
    matBCurrElem.setPosition(matBLeftEdgeX, matBTopEdgeY);
    matACurrRowRect.setPosition(matALeftEdgeX, matATopEdgeY);
    matBCurrColRect.setPosition(matBLeftEdgeX, matBTopEdgeY);
    matOutCurrElem.setPosition(matOutLeftEdgeX, 50);
    matACurrElem.setFillColor(sf::Color(51, 153, 255));
    matBCurrElem.setFillColor(sf::Color(51, 153, 255));
    matACurrRowRect.setFillColor(sf::Color(51, 153, 255));
    matBCurrColRect.setFillColor(sf::Color(51, 153, 255));
    matOutCurrElem.setFillColor(sf::Color::Red);

    // Set initial positions for current tile in tiling mode
    matATileRect.setPosition(matALeftEdgeX, matATopEdgeY);
    matBTileRect.setPosition(matBLeftEdgeX, matBTopEdgeY);
    matOutTileRect.setPosition(matOutLeftEdgeX, 50);
    matATileRect.setFillColor(sf::Color(51, 153, 255));
    matBTileRect.setFillColor(sf::Color(51, 153, 255));
    matOutTileRect.setFillColor(sf::Color::Red);

    int iteration = 0;
    sf::Clock iterationTimer;
    iterationTimer.restart();

    // Indices for naive CUDA mode
    int matACurrRow = 0;
    int matBCurrCol = 0;

    // Position tracking for CPU mode
    float matACurrElemX = matALeftEdgeX;
    float matACurrElemY = matATopEdgeY;
    float matBCurrElemX = matBLeftEdgeX;
    float matBCurrElemY = matBTopEdgeY;
    float matOutCurrElemX = matOutLeftEdgeX;
    float matOutCurrElemY = 50;

    std::string str;

    // --- Initial calculation to display the result of the first iteration depending on the mode ---

    // CPU mode
    if (mode == 0)
    {
        matOut[0][0] = matA[0][0] * matB[0][0];
        matOutNumberTexts[0].setString(std::to_string(matOut[0][0]));
        str = matOutNumberTexts[0].getString();
        if (str.length() == 2)
        {
            matOutNumberTexts[0].setPosition(matOutLeftEdgeX + 6.5, matOutNumberTexts[0].getPosition().y);
        }
    }
    // Naive CUDA mode
    else if (mode == 1)
    {
        for (int i = 0; i < matrixHeight; i++)
        {
            matOut[0][0] += matA[0][i] * matB[i][0];
        }
        matOutNumberTexts[0].setString(std::to_string(matOut[0][0]));
        str = matOutNumberTexts[0].getString();
        if (str.length() == 2)
        {
            matOutNumberTexts[0].setPosition(matOutLeftEdgeX + 6.5, matOutNumberTexts[0].getPosition().y);
        }
        else if (str.length() == 3)
        {
            matOutNumberTexts[0].setPosition(matOutLeftEdgeX + 3.5, matOutNumberTexts[0].getPosition().y);
        }
        else if (str.length() == 4)
        {
            matOutNumberTexts[0].setPosition(matOutLeftEdgeX + 0.5, matOutNumberTexts[0].getPosition().y);
        }
    }
    // Tiling mode
    else if (mode == 2)
    {
        // Calculate first 4x4 tile of the output matrix
        for (int i = 0; i < matrixHeight / 8; i++)
        {
            for (int j = 0; j < matrixWidth / 8; j++)
            {
                for (int k = 0; k < matrixWidth / 8; k++)
                {
                    matOut[i][j] += matA[i][k] * matB[k][j];
                }
                matOutNumberTexts[matrixHeight * i + j].setString(std::to_string(matOut[i][j]));
                str = matOutNumberTexts[matrixHeight * i + j].getString();
                if (str.length() == 2)
                {
                    matOutNumberTexts[matrixHeight * i + j].setPosition(matOutLeftEdgeX + 6.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[matrixHeight * i + j].getPosition().y);
                }
                else if (str.length() == 3)
                {
                    matOutNumberTexts[matrixHeight * i + j].setPosition(matOutLeftEdgeX + 3.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[matrixHeight * i + j].getPosition().y);
                }
                else if (str.length() == 4)
                {
                    matOutNumberTexts[matrixHeight * i + j].setPosition(matOutLeftEdgeX + 0.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[matrixHeight * i + j].getPosition().y);
                }
            }
        }
    }

    // Indices for iteration control for CPU mode
    int i = 0; // Row index for the output matrix
    int j = 0; // Column index for the output matrix
    int k = 0; // Inner dimension (columns of matrix A) index

    // Tile indices for tiling mode
    int matACurrTileX = 0;
    int matACurrTileY = 0;
    int matBCurrTileX = 0;
    int matBCurrTileY = 0;
    int matOutCurrTileX = 0;
    int matOutCurrTileY = 0;

    // Determine total iterations to display and delay for visualization based on the selected mode
    int totalIterations;
    int millisToWait = 0;

    if (mode == 1)
    {
        totalIterations = 32 * 32;
        millisToWait = 20;
    }
    else if (mode == 0)
    {
        totalIterations = 32 * 32 * 32;
        millisToWait = 20;
    }
    else if (mode == 2)
    {
        totalIterations = 8 * 8 * 8;
        millisToWait = 250;
    }

    // Flag for tiling mode to track if data has been copied to shared memory in the visualization
    bool copiedToSharedMem = false;

    // Main Rendering Loop
    while (window.isOpen())
    {
        // Quit application if ESC is pressed
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

        // Check if enough time has passed for the next calculation step
        if (iterationTimer.getElapsedTime().asMilliseconds() > millisToWait)
        {
            iterationTimer.restart();
            
            // Continue the visualization if not all iterations are complete
            if (iteration < totalIterations - 1)
            {
                // Naive CUDA mode
                if (mode == 1)
                {
                    ++iteration;
                    // Move to the next row of matrix A and reset matrix B to the first column after a full row has been processed
                    if (iteration % 32 == 0)
                    {
                        matACurrRow++;
                        matBCurrCol = 0;
                    }
                    // Move to the next column of matrix B
                    else
                    {
                        matBCurrCol++;
                    }
                    // Calculate the complete dot product for a single element of the output matrix
                    for (int i = 0; i < matrixHeight; i++)
                    {
                        matOut[matACurrRow][matBCurrCol] += matA[matACurrRow][i] * matB[i][matBCurrCol];
                    }

                    // Update the text for the new output element and adjust the text if a new digit has been added
                    matOutNumberTexts[32 * matACurrRow + matBCurrCol].setString(std::to_string(matOut[matACurrRow][matBCurrCol]));
                    str = std::to_string(matOut[matACurrRow][matBCurrCol]);
                    if (str.length() == 2)
                    {
                        matOutNumberTexts[32 * matACurrRow + matBCurrCol].setPosition(matOutLeftEdgeX + 6.5 + matOutHeight / 32 * ((32 * matACurrRow + matBCurrCol) % 32), matOutNumberTexts[32 * matACurrRow + matBCurrCol].getPosition().y);
                    }
                    else if (str.length() == 3)
                    {
                        matOutNumberTexts[32 * matACurrRow + matBCurrCol].setPosition(matOutLeftEdgeX + 3.5 + matOutHeight / 32 * ((32 * matACurrRow + matBCurrCol) % 32), matOutNumberTexts[32 * matACurrRow + matBCurrCol].getPosition().y);
                    }
                    else if (str.length() == 4)
                    {
                        matOutNumberTexts[32 * matACurrRow + matBCurrCol].setPosition(matOutLeftEdgeX + 0.5 + matOutHeight / 32 * ((32 * matACurrRow + matBCurrCol) % 32), matOutNumberTexts[32 * matACurrRow + matBCurrCol].getPosition().y);
                    }
                }
                // CPU mode
                else if (mode == 0)
                {
                    ++iteration;

                    // Move to the next row of the output matrix after a full row has been calculated
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
                    // Move to the next column of the output matrix after a full dot product is complete
                    else if (iteration % 32 == 0)
                    {
                        j++;
                        k = 0;
                        matACurrElemX = matALeftEdgeX;
                        matBCurrElemX += matBHeight / 32;
                        matBCurrElemY = matBTopEdgeY;
                        matOutCurrElemX += matOutHeight / 32;
                    }
                    // Move to the next step of the dot product for the current output matrix element
                    else
                    {
                        k++;
                        matACurrElemX += matAHeight / 32;
                        matBCurrElemY += matBHeight / 32;
                    }

                    // Perform the multiplication and accumulation 
                    matOut[i][j] += matA[i][k] * matB[k][j];

                    // Update the changed output matrix element and adjust the position if a new digit has been added
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
                // Tiling mode
                else if (mode == 2)
                {
                    // If the tile was copied to shared memory, then the computation for that tile
                    // was also completed, so move on to the next tile
                    if (copiedToSharedMem)
                    {
                        iteration++;
                        copiedToSharedMem = false; // Next step will be the copy and calculation phase

                        // Move to the next row of output tiles after a full row of block multiplication is complete
                        if (iteration % (8 * 8) == 0)
                        {
                            matOutCurrTileX = 0;
                            matOutCurrTileY++;
                            matACurrTileX = 0;
                            matACurrTileY++;
                            matBCurrTileX = 0;
                            matBCurrTileY = 0;
                        }
                        // Move to the next output column tile after a full column of block multiplications for the current tile is complete
                        else if (iteration % 8 == 0)
                        {
                            matOutCurrTileX++;
                            matACurrTileX = 0;
                            matBCurrTileX++;
                            matBCurrTileY = 0;
                        }
                        // Move to the next tile block in matrices A and B for the same output tile
                        else
                        {
                            matACurrTileX++;
                            matBCurrTileY++;
                        }
                    }
                    // The current phase is the copy to shared memory and calculation phase
                    else
                    {
                        copiedToSharedMem = true;

                        // Updated the text in the shared memory grids
                        for (int i = 0; i < 4 * 4; i++)
                        {
                            matASharedMemNumberTexts[i].setString(std::to_string(matA[matACurrTileY * 4 + i / 4][matACurrTileX * 4 + i % 4]));
                            matBSharedMemNumberTexts[i].setString(std::to_string(matB[matBCurrTileY * 4 + i / 4][matBCurrTileX * 4 + i % 4]));
                        }

                        // Perform block multiplication and update the result
                        for (int i = 0 + 4 * matOutCurrTileY; i < matrixHeight / 8 + 4 * matOutCurrTileY; i++)
                        {
                            for (int j = 0 + 4 * matOutCurrTileX; j < matrixWidth / 8 + 4 * matOutCurrTileX; j++)
                            {
                                for (int k = 0 + 4 * matACurrTileX; k < matrixWidth / 8 + 4 * matACurrTileX; k++)
                                {
                                    matOut[i][j] += matA[i][k] * matB[k][j];
                                }

                                // Update and reposition the text for the current output element
                                matOutNumberTexts[matrixHeight * i + j].setString(std::to_string(matOut[i][j]));
                                str = matOutNumberTexts[matrixHeight * i + j].getString();
                                if (str.length() == 2)
                                {
                                    matOutNumberTexts[matrixHeight * i + j].setPosition(matOutLeftEdgeX + 6.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[matrixHeight * i + j].getPosition().y);
                                }
                                else if (str.length() == 3)
                                {
                                    matOutNumberTexts[matrixHeight * i + j].setPosition(matOutLeftEdgeX + 3.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[matrixHeight * i + j].getPosition().y);
                                }
                                else if (str.length() == 4)
                                {
                                    matOutNumberTexts[matrixHeight * i + j].setPosition(matOutLeftEdgeX + 0.5 + matOutHeight / 32 * ((32 * i + j) % 32), matOutNumberTexts[matrixHeight * i + j].getPosition().y);
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Update Visualization Element Positions ---

        // Naive CUDA mode
        if (mode == 1)
        {
            matACurrRowRect.setPosition(matALeftEdgeX, matATopEdgeY + matAHeight / 32.0f * matACurrRow);
            matBCurrColRect.setPosition(matBLeftEdgeX + matBHeight / 32.0f * matBCurrCol, matBTopEdgeY);
            matOutCurrElem.setPosition(matOutLeftEdgeX + matOutHeight / 32.0f * matBCurrCol, 50 + matOutHeight / 32.0f * matACurrRow);
        }
        // CPU mode
        else if (mode == 0)
        {
            matACurrElem.setPosition(matACurrElemX, matACurrElemY);
            matBCurrElem.setPosition(matBCurrElemX, matBCurrElemY);
            matOutCurrElem.setPosition(matOutCurrElemX, matOutCurrElemY);
        }
        // Tiling mode
        else if (mode == 2)
        {
            // When not copying a tile to shared memory, highlight the next tiles in matrices A and B
            if (!copiedToSharedMem)
            {
                matATileRect.setPosition(matALeftEdgeX + matACurrTileX * matAHeight / 8.0f, matATopEdgeY + matACurrTileY * matAHeight / 8.0f);
                matBTileRect.setPosition(matBLeftEdgeX + matBCurrTileX * matAHeight / 8.0f, matBTopEdgeY + matBCurrTileY * matAHeight / 8.0f);
                matOutTileRect.setPosition(matOutLeftEdgeX + matOutCurrTileX * matOutHeight / 8.0f, 50 + matOutCurrTileY * matOutHeight / 8.0f);
            }
            // During the copy and computation phase, highlight the tile in the shared memory grid
            else
            {
                matATileRect.setPosition(matASharedMemLeftEdgeX, matASharedMemTopEdgeY);
                matBTileRect.setPosition(matBSharedMemLeftEdgeX, matBSharedMemTopEdgeY);
            }
        }

        // --- SFML Drawing ---

        window.clear(sf::Color::White);

        // Draw matrix outlines
        window.draw(matAOutline);
        window.draw(matBOutline);
        window.draw(matOutOutline);
        // Naive CUDA mode
        if (mode == 1)
        {
            window.draw(matACurrRowRect);
            window.draw(matBCurrColRect);
            window.draw(matOutCurrElem);
        }
        // CPU mode
        else if (mode == 0)
        {
            window.draw(matACurrElem);
            window.draw(matBCurrElem);
            window.draw(matOutCurrElem);
        }
        // Tiling mode
        else if (mode == 2)
        {
            window.draw(matASharedMem);
            window.draw(matBSharedMem);
            window.draw(matATileRect);
            window.draw(matBTileRect);
            window.draw(matOutTileRect);
            window.draw(matASharedMemLines);
            window.draw(matBSharedMemLines);
        }

        // Draw matrix grid lines
        window.draw(matALines);
        window.draw(matBLines);
        window.draw(matOutLines);
        window.draw(multiplySign);
        window.draw(equalSign);

        // Draw all matrix element numbers
        for (int i = 0; i < 32 * 32; i++)
        {
            window.draw(matANumberTexts[i]);
            window.draw(matBNumberTexts[i]);
            window.draw(matOutNumberTexts[i]);
        }

        // Draw shared memory numbers only in Tiling mode
        if (mode == 2)
        {
            for (int i = 0; i < 4 * 4; i++)
            {
                window.draw(matASharedMemNumberTexts[i]);
                window.draw(matBSharedMemNumberTexts[i]);
            }
        }
        window.display();
    }

    return 0;
}