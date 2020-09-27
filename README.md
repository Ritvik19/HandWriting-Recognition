# HandWriting Recognition
       
### Lenses:

        Train a model

        usage: train.py [-h] [-a algo] [-d data] [-p [problem]] [-t [type]]

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:

        -a algo, --algo algo  algorithm to run (lens-<lesn-name>)
        -d data, --dataset data dataset to train on (<lens-name>)
        -p [problem], --problem [problem] type of problem (mc)
        -t [type], --type [type] type of dataset (img)

* lens-digi: 
    
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9
* lens-alpha
     
        A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b d e f g h n q r t
* lens-alnum: 
            
        0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b d e f g h n q r t
* lens-kdigi:
        
        ೦, ೧, ೨, ೩, ೪, ೫, ೬, ೭, ೮, ೯
* lens-ddigi:
        
        ०, १, २, ३, ४, ५, ६, ७, ८, ९
* lens-maths:
        
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, (, ), +, ÷ , =, ×, - 

### Applications

1.  Scratchpad        
                
        Scratchpad Application to play around with the lenses

        usage: Scratchpad.py [-h] -l LENS

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:
        -l LENS, --lens LENS  name of the lens

2. Basic Calculator

        Simple Calculator Application that works on handwriting input

        usage: BasicCalc.py


3. [Sudoku-Lens](https://github.com/Ritvik19/Sudoku-Lens)

        Application that solves Sudoku Quizzes using neural networks

4. OCR

        OCR Application to read text from an image

        usage: OCR.py [-h] -p PATH

        optional arguments:
        -h, --help            show this help message and exit

        required arguments:
        -p PATH, --path PATH  path of the image