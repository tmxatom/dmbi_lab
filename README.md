# Simple Red Calculator

A straightforward client-side calculator built with HTML, CSS, and JavaScript, featuring a distinct red theme for its operations.

## Features

*   Basic arithmetic operations: addition, subtraction, multiplication, division.
*   Clear (C) and Delete (DEL) functionality.
*   Decimal point support.
*   Responsive design for various screen sizes.
*   Keyboard support for numbers, operators, Enter, Backspace, and Escape.
*   Vibrant red theme for operator and equals buttons.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd simple-red-calculator
    ```
    (For this standalone project, you would simply create the files in a directory.)

2.  **Install dependencies:**
    This project is primarily client-side and doesn't have runtime dependencies that require `npm install` for the calculator itself. However, a `package.json` is provided for convenient local serving using `live-server`.

    ```bash
    npm install
    ```
    This will install `live-server` as a development dependency.

## Setup Commands

No specific setup commands are required beyond installing `live-server` if you wish to use the `npm run serve` command.

## How to Run the Project

You have two options to run the calculator:

### Option 1: Open `index.html` directly

Simply open the `index.html` file in your web browser. Drag and drop the file into your browser's window, or navigate to its path using `File -> Open File`.

### Option 2: Use a Local HTTP Server (Recommended for development)

Using a local HTTP server like `live-server` provides a better development experience (e.g., live reloading).

1.  **Ensure `live-server` is installed (as per Installation step 2):**
    ```bash
    npm install
    ```

2.  **Start the server:**
    ```bash
    npm run serve
    ```
    This will open the calculator in your default web browser, typically at `http://127.0.0.1:8080`.

## How to Run Tests

No tests are included or required for this simple project.

## Environment Variables

No environment variables are needed for this client-side project.
