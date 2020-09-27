import subprocess, os

tests = [
    ('lens-digi', 'digi', 'mc', 'img'),
    ('lens-alpha', 'alpha', 'mc', 'img'),
    ('lens-alnum', 'alnum', 'mc', 'img'),
    ('lens-kdigi', 'kdigi', 'mc', 'img'),
    ('lens-ddigi', 'ddigi', 'mc', 'img'),
    ('lens-maths', 'maths', 'mc', 'img'),
]

for model, data, problem, dtype in tests:
    exit_code = subprocess.call([
        "D:\\Users\\Ritvik\\Anaconda3\\envs\\ailab\\python.exe",
        os.path.join(os.getcwd(), "train.py"), 
        f"-a {model}", 
        f"-d {data}",
        f"-p {problem}",
        f"-t {dtype}"
    ])
    subprocess.call([
        "D:\\Users\\Ritvik\\Anaconda3\\envs\\ailab\\python.exe",
        os.path.join(os.getcwd(), "performance_report.py")
    ])
    if exit_code == 0:
        print('\ntrained', model, '\n\n')
    else:
        print('\nfailed', model, '\n\n')
