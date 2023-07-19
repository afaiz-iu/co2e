from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

gpu_map = {
    'A100': {'tdp': 0.4, 'peak': 312},
    'V100': {'tdp': 0.3, 'peak': 125}
}

# regression coefficients basis observed Megatron scaling for throughput
coeff_tensor = np.array([-8.82079068e-20,  1.68591116e-09,  1.33954735e+02])
coeff_pipe = np.array([-5.60233749e-23,  8.45435587e-11,  1.34546129e+02])
coeff_gpu = np.array([-2.12910565e-21,  4.39684339e-09,  7.99173057e+02])
coeff_batch = np.array([-4.29439186e-01,  5.21376002e+01,  1.43737095e+03])

func_tensor = np.poly1d(coeff_tensor)
func_pipe = np.poly1d(coeff_pipe)
func_gpu = np.poly1d(coeff_gpu)
func_batch = np.poly1d(coeff_batch)

def get_ptd(P, node_size, gpu_cap, gpu_type, gpu_mem):
    p_b = P/1e9

    # model parallel size
    if p_b < node_size*gpu_cap:
        p_size = 1
        t_size = int(np.ceil(p_b/gpu_cap))
    else:
        t_size = node_size
        p_size = int(np.ceil(p_b/(node_size*gpu_cap)))

    model_size = t_size * p_size
    
    # number of gpus estimate
    num_gpu = np.round(func_gpu(P)/model_size)*model_size
    if 'V100' in gpu_type:
        num_gpu = np.round(num_gpu * 2.5)
        
    if gpu_mem == 40:
        num_gpu *= 2
    
    d_size = num_gpu/model_size
    #estimated batch size
    if p_size == 1:
        batch_size = 512
    else:
        batch_size = np.round(func_batch(p_size)/8)*8
        if batch_size < num_gpu:
            batch_size = num_gpu

    return p_size, t_size, d_size, num_gpu, batch_size

def get_throughput(t_size, p_size, node_size, P, gpu_type, rel_thru):
    #intra model condition
    if (t_size <= node_size and p_size == 1):
        X = func_tensor(P)
    # inter model     
    else:
        X = func_pipe(P)

    if 'V100' in gpu_type:
        X_new = X -  X*rel_thru
        peak_new = X_new /312
        X_scaled = peak_new*125
    else:
        X_scaled = X    
    
    return X_scaled

def get_train_time(model_type, tokens, P, n, X):
    flop_per_parameter = 6
    if 'T5' in model_type:
        flop_per_parameter = 3
    
    total_compute = P*tokens*flop_per_parameter
    total_compute_per_sec = n*X*1e12
    
    train_sec = total_compute / total_compute_per_sec
    
    return train_sec, total_compute

def get_co2e(gpu_tdp, train_time, region_co2, pue, n):
    co2_gpu = gpu_tdp * train_time * region_co2 * pue
    co2_gross = co2_gpu*n
    return co2_gross

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_type = request.form.get('model_type', 'GPT')
        tokens = request.form.get('tokens', 'NA')
        if tokens.lower() != 'na':
            tokens = int(float(tokens))
        V = int(request.form.get('V', 0))
        s = int(request.form.get('s', 0))
        h = int(request.form.get('h', 0))
        a = int(request.form.get('a', 0))
        l = int(request.form.get('l', 0))
        gpu_type = request.form.get('gpu_type', 'A100')
        node_size = int(request.form.get('node_size', 8))
        gpu_mem = int(request.form.get('gpu_mem', 80))
        region_co2 = request.form.get('datacenter_co2', 0.429)
        pue = request.form.get('pue', 1.1)

        parameter_str = request.form.get('parameter', 'NA')
        if parameter_str.lower() == 'na':
            # Calculate total parameters based on other values
            P = 12 * l * (h**2) * (1 + (13 / (12 * h)) + ((V + s) / (12 * l * h)))
        else:
            try:
                P = int(float(parameter_str))  # scientific to integer
            except ValueError:
                P = 0

        gpu_cap = 0.03*gpu_mem
        node_cap = node_size*gpu_cap
        gpu_tdp = gpu_map[gpu_type]['tdp']
        gpu_peak = gpu_map[gpu_type]['peak']
        rel_thru = 7.76 / 33.46

        p_size, t_size, d_size, num_gpu, batch_size = get_ptd(P, node_size, gpu_cap, gpu_type, gpu_mem)

        X = get_throughput(t_size, p_size, node_size, P, gpu_type, rel_thru)

        train_sec, total_compute  = get_train_time(model_type, tokens, P, num_gpu, X)
        train_hour = np.round(train_sec/3600)
        train_day = np.ceil(train_sec/86400)


        co2e_gross = get_co2e(float(gpu_tdp), float(train_hour), float(region_co2), float(pue), float(num_gpu))

        return render_template('results.html', num_gpu=int(num_gpu), p_size=p_size, t_size=t_size, d_size=int(d_size),
                               batch_size=int(batch_size), total_compute=total_compute, X=X, gpu_peak=gpu_peak,
                               train_day=int(train_day), co2e_gross=co2e_gross/1000)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
