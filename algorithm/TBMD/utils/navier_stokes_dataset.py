import torch
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from timeit import default_timer


class GaussianRF:
    """
    Gaussian Random Field Generator

    Generates random fields with Gaussian statistics and prescribed spatial correlation structure.
    
    Parameters
    ----------
    dim : int
        Dimensionality of the random field (1, 2, or 3)
    size : int
        Size of the domain (must be power of 2)
    alpha : float, optional
        Smoothness parameter (default: 2)
    tau : float, optional
        Correlation length parameter (default: 3)
    sigma : float, optional
        Standard deviation parameter (default: computed from tau and alpha)
    boundary : str, optional
        Boundary condition type (default: "periodic")
    device : torch.device, optional
        Device for computation (default: None)
    """

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):
        """
        Generate N samples from the random field
        
        Parameters
        ----------
        N : int
            Number of samples to generate
            
        Returns
        -------
        torch.Tensor
            Tensor of shape (N, *self.size) containing the samples
        """
        coeff = torch.randn(N, *self.size, 2, device=self.device)

        coeff[...,0] = self.sqrt_eig*coeff[...,0]
        coeff[...,1] = self.sqrt_eig*coeff[...,1]

        if self.dim == 1:
            complex_coeff = torch.complex(coeff[..., 0], coeff[..., 1])
            u = torch.fft.ifft(complex_coeff, dim=-1)
            u = u.real
        elif self.dim == 2:
            complex_coeff = torch.complex(coeff[..., 0], coeff[..., 1])
            u = torch.fft.ifft2(complex_coeff, dim=(-2, -1))
            u = u.real
        elif self.dim == 3:
            complex_coeff = torch.complex(coeff[..., 0], coeff[..., 1])
            u = torch.fft.ifftn(complex_coeff, dim=(-3, -2, -1))
            u = u.real

        return u


def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):
    """
    Solve the 2D Navier-Stokes equations in vorticity form
    
    Parameters
    ----------
    w0 : torch.Tensor
        Initial vorticity field
    f : torch.Tensor
        Forcing term
    visc : float
        Viscosity (1/Reynolds number)
    T : float
        Final simulation time
    delta_t : float, optional
        Time step for numerical integration (default: 1e-4)
    record_steps : int, optional
        Number of snapshots to record (default: 1)
        
    Returns
    -------
    sol : torch.Tensor
        Solution tensor of shape (*w0.size(), record_steps)
    sol_t : torch.Tensor
        Time points corresponding to recorded solutions
    """
    # Grid size - must be power of 2
    N = w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N/2.0)

    # Number of steps to final time
    steps = math.ceil(T/delta_t)

    # Подготовка FFT плоскости для вычислений
    # Используем fft2 вместо rfft2 для сохранения полного спектра
    w_h_complex = torch.fft.fft2(w0)
    # Разделяем на реальную и мнимую части для совместимости с оригинальным кодом
    w_h = torch.stack((w_h_complex.real, w_h_complex.imag), dim=-1)

    # Forcing to Fourier space
    f_h_complex = torch.fft.fft2(f)
    f_h = torch.stack((f_h_complex.real, f_h_complex.imag), dim=-1)

    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device), 
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    
    # Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    
    # Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max, 
                                               torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0
    for j in range(steps):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[...,0] = psi_h[...,0]/lap
        psi_h[...,1] = psi_h[...,1]/lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[...,0].clone()
        q[...,0] = -2*math.pi*k_y*q[...,1]
        q[...,1] = 2*math.pi*k_y*temp
        
        # Преобразуем в комплексный тензор для нового API
        q_complex = torch.complex(q[..., 0], q[..., 1])
        q = torch.fft.ifft2(q_complex).real  # Возвращаем только реальную часть

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[...,0].clone()
        v[...,0] = 2*math.pi*k_x*v[...,1]
        v[...,1] = -2*math.pi*k_x*temp
        
        v_complex = torch.complex(v[..., 0], v[..., 1])
        v = torch.fft.ifft2(v_complex).real  # Возвращаем только реальную часть

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[...,0].clone()
        w_x[...,0] = -2*math.pi*k_x*w_x[...,1]
        w_x[...,1] = 2*math.pi*k_x*temp
        
        w_x_complex = torch.complex(w_x[..., 0], w_x[..., 1])
        w_x = torch.fft.ifft2(w_x_complex).real  # Возвращаем только реальную часть

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[...,0].clone()
        w_y[...,0] = -2*math.pi*k_y*w_y[...,1]
        w_y[...,1] = 2*math.pi*k_y*temp
        
        w_y_complex = torch.complex(w_y[..., 0], w_y[..., 1])
        w_y = torch.fft.ifft2(w_y_complex).real  # Возвращаем только реальную часть

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h_complex = torch.fft.fft2(q*w_x + v*w_y)
        F_h = torch.stack((F_h_complex.real, F_h_complex.imag), dim=-1)

        # Dealias
        F_h[...,0] = dealias * F_h[...,0]
        F_h[...,1] = dealias * F_h[...,1]

        # Cranck-Nicholson update
        w_h[...,0] = (-delta_t*F_h[...,0] + delta_t*f_h[...,0] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,0])/(1.0 + 0.5*delta_t*visc*lap)
        w_h[...,1] = (-delta_t*F_h[...,1] + delta_t*f_h[...,1] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,1])/(1.0 + 0.5*delta_t*visc*lap)

        # Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            # Solution in physical space
            w_h_complex = torch.complex(w_h[..., 0], w_h[..., 1])
            w = torch.fft.ifft2(w_h_complex).real  # Возвращаем только реальную часть

            # Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t


def generate_navier_stokes_dataset(resolution=256, num_samples=20, batch_size=20, record_steps=200, save_path='ns_data.mat'):
    """
    Generate a dataset of Navier-Stokes solutions with random initial conditions
    
    Parameters
    ----------
    resolution : int, optional
        Resolution of the spatial grid (default: 256)
    num_samples : int, optional
        Number of total samples to generate (default: 20)
    batch_size : int, optional
        Batch size for parallel computation (default: 20)
    record_steps : int, optional
        Number of temporal snapshots to record (default: 200)
    save_path : str, optional
        Path to save the generated data (default: 'ns_data.mat')
        
    Returns
    -------
    None
        Data is saved to the specified file
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up grid resolution
    s = resolution
    
    # Set up 2D Gaussian Random Field with covariance parameters
    GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)
    
    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, 1, s+1, device=device)
    t = t[0:-1]
    
    # Добавляем параметр indexing='ij' для исправления предупреждения
    X, Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
    
    # Initialize arrays for inputs and solutions
    a = torch.zeros(num_samples, s, s)  # Initial conditions
    u = torch.zeros(num_samples, s, s, record_steps)  # Solutions at different time steps
    
    # Process data in batches for efficiency
    c = 0
    t0 = default_timer()
    for j in range(num_samples // batch_size):
        # Sample random initial fields
        w0 = GRF.sample(batch_size)
        
        # Solve Navier-Stokes equations
        sol, sol_t = navier_stokes_2d(w0, f, 1e-3, 50.0, 1e-4, record_steps)
        
        # Store data
        a[c:(c+batch_size),...] = w0
        u[c:(c+batch_size),...] = sol
        
        c += batch_size
        t1 = default_timer()
        print(f"Batch {j+1}/{num_samples // batch_size} completed. Generated {c}/{num_samples} samples. Time: {t1-t0:.2f}s")
    
    # Save data to file
    print(f"Saving data to {save_path}...")
    scipy.io.savemat(save_path, mdict={'a': a.cpu().numpy(), 'u': u.cpu().numpy(), 't': sol_t.cpu().numpy()})
    print(f"Data generation completed. Total time: {default_timer()-t0:.2f}s")

