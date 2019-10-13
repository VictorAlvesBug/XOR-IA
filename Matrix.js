class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;

        this.data = [];

        for (let i = 0; i < rows; i++) {
            let arr = []
            for (let j = 0; j < cols; j++) {
                arr.push(0)
            }
            this.data.push(arr);
        }
    }

    // Fun��es Diversas

    static arrayToMatrix(arr) {
        let matrix = new Matrix(arr.length, 1);
        matrix.map((element, i, j) => {
            return arr[i];
        });

        return matrix;
    }

    static MatrixToArray(obj) {
        let arr = []
        obj.map((elm, i, j) => {
            arr.push(elm);

        })

        return arr;
    }

    print() {
        console.table(this.data);
    }

    randomize() {
        this.map((element, i, j) => {
            return Math.random() * 2 - 1;
        });
        //return this;
    }

    map(func) {
        this.data = this.data.map((arr, i) => {
            return arr.map((num, j) => {
                return func(num, i, j);
            })
        })

        return this;
    }

    static map(A, func) {
        let matrix = new Matrix(A.rows, A.cols);

        matrix.data = A.data.map((arr, i) => {
            return arr.map((num, j) => {
                return func(num, i, j);
            })
        })

        return matrix;
    }

    static transpose(A) {
        var matrix = new Matrix(A.cols, A.rows);
        matrix.map((element, i, j) => {
            return A.data[j][i];
        });

        return matrix;
    }

    // Opera��es Est�ticas Matriz x Escalar

    static escalar_add(A, escalar) {
        var matrix = new Matrix(A.rows, A.cols);
        matrix.map((element, i, j) => {
            return A.data[i][j] + escalar;
        });

        return matrix;
    }

    static escalar_subtract(A, escalar) {
        var matrix = new Matrix(A.rows, A.cols);
        matrix.map((element, i, j) => {
            return A.data[i][j] - escalar;
        });

        return matrix;
    }

    static escalar_multiply(A, escalar) {
        var matrix = new Matrix(A.rows, A.cols);
        matrix.map((element, i, j) => {
            return A.data[i][j] * escalar;
        });

        return matrix;
    }

    // Opera��es Est�ticas Matriz x Matriz

    static add(A, B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw new Error('Matrizes n�o compat�veis para efetuar adi��o.');
            return null;
        }

        var matrix = new Matrix(A.rows, A.cols);
        matrix.map((element, i, j) => {
            return A.data[i][j] + B.data[i][j];
        });

        return matrix;
    }

    static subtract(A, B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw new Error('Matrizes n�o compat�veis para efetuar subtra��o.');
            return null;
        }

        var matrix = new Matrix(A.rows, A.cols);
        matrix.map((element, i, j) => {
            return A.data[i][j] - B.data[i][j];
        });

        return matrix;
    }

    static multiply(A, B) {
        if (A.cols != B.rows) {
            throw new Error('Matrizes n�o compat�veis para efetuar multiplica��o.');
            return null;
        }

        var matrix = new Matrix(A.rows, B.cols);
        matrix.map((element, i, j) => {
            let sum = 0;
            for (let k = 0; k < A.cols; k++) {
                sum += A.data[i][k] * B.data[k][j];
            }
            return sum;
        });

        return matrix;
    }

    static hadamard(A, B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw new Error('Matrizes n�o compat�veis para efetuar produto Hadamard.');
            return null;
        }

        var matrix = new Matrix(A.rows, A.cols);
        matrix.map((element, i, j) => {
            return A.data[i][j] * B.data[i][j];
        });

        return matrix;
    }
}