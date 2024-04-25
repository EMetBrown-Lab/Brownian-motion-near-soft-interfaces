"""
Wrapping functions to compute observables from a Brownian trajectories.
"""


# =============================================================================
# Importations
# =============================================================================
import numpy as np
from StochasticForceInference.fun_SFI import Compute_diffusion
from StochasticForceInference.StochasticForceInference import *
import inspect
from scipy.io import loadmat


# =============================================================================
# Open Files
# =============================================================================

def open_xyz_mat(pathname, upward=False, version='new'):
    data = loadmat(pathname, squeeze_me=True)
    if version == 'new':
        raw_data = np.zeros((len(data['x']), 3))
        raw_data[:,0] = data['x']
        raw_data[:,1] = data['y']
        raw_data[:,2] = data['z']
    elif version == 'old':
        raw_data = data["data"][:, 0:3]
    else:
        print("Unknown version. Only values accepted = 'new' or 'old'.")
        print("Try again.")
        version = str(input("Data version? Enter old or new: "))
        raw_data = open_xyz_mat(pathname, upward=upward, version=version)
    if upward:
        raw_data[:,2] = - raw_data[:,2]
    del data
    return raw_data
    
    
# =============================================================================
# Aux
# =============================================================================

def remove_end_zeros(data):
    try:
        ind = list(data[:,0]).index(0)
        data = data[:ind,:]
    except ValueError:
        print('No pb. No zero in raw data.')
    return data

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    res = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    if len(res) == 0:
        print('No variable found!')
        res = ''
    elif len(res) == 1:
        res = res[0]
    else:
        print(res)
        which = int(input('Which variable do you want? (Python index)'))
        res = res[which]
    return res


# =============================================================================
# Operations on arrays
# =============================================================================

def movmin(datas, k):
    result = np.empty_like(datas)
    start_pt = 0
    end_pt = int(np.ceil(k / 2))

    for i in range(len(datas)):
        if i < int(np.ceil(k / 2)):
            start_pt = 0
        if i > len(datas) - int(np.ceil(k / 2)):
            end_pt = len(datas)
        result[i] = np.nanmin(datas[start_pt:end_pt])
        start_pt += 1
        end_pt += 1

    return result

def movmean(datas, k):
    result = np.empty_like(datas)
    start_pt = 0
    end_pt = int(np.ceil(k / 2))

    for i in range(len(datas)):
        if i < int(np.ceil(k / 2)):
            start_pt = 0
        if i > len(datas) - int(np.ceil(k / 2)):
            end_pt = len(datas)
        result[i] = np.mean(datas[start_pt:end_pt])
        start_pt += 1
        end_pt += 1

    return result

def remove_ending_zeros(data):
    # Remove zeros at the end of the trajectory:
    try:
        ind = list(data[:,0]).index(0)
    except ValueError:
        # print('No pb. No zero in raw data.')
        ind = len(data[:,0])
    res = data[:ind,:]
    return res


# =============================================================================
# PDF
# =============================================================================

def pdf(data, bins=10, density=True, range=None):
    pdf, bins_edge = np.histogram(
        data, 
        bins=bins, 
        density=density, 
        range=range, 
    )
    bins_center = (bins_edge[0:-1] + bins_edge[1:]) / 2
    return pdf, bins_center

def logarithmic_hist(data, begin, stop, num=50, base=2, density=True):
    if begin == 0:
        beg = stop / num
        bins = np.logspace(
            np.log(beg) / np.log(base), 
            np.log(stop) / np.log(base), 
            num - 1, 
            base=base, 
        )
        widths = bins[1:] - bins[:-1]
        bins = np.cumsum(widths[::-1])
        bins = np.concatenate(([0], bins))
        widths = bins[1:] - bins[:-1]
    else:
        bins = np.logspace(
            np.log(begin) / np.log(base), 
            np.log(stop) / np.log(base), 
            num, 
            base=base, 
        )
        widths = bins[1:] - bins[:-1]
    hist, a = np.histogram(data, bins=bins, density=density)
    # normalize by bin width
    bins_center = (bins[1:] + bins[:-1]) / 2
    return bins_center, widths, hist


# =============================================================================
# Equilibrium probability distribution function and force
# =============================================================================

def PeqFeq(z_dedrift, n_noisy, min_distance, max_distance, n_bins):
    """
    Compute the equilibrium probability density function and force, as a 
    function to the distance to the wall.
    The trajectory is noised n_noisy times. The final result is the average 
    of every computations.
    
    Parameters
    ----------
    z_dedrift : array
        z-trajectory of the particle.
    n_noisy : int 
        Number of iterations.
    min_distance, max_distance : floats
        Spatial range.
    n_bins : int
        Number of bins. 
    

    Returns
    -------
    P_eq, F_eq : two dict
        The dictionnaries contain the data corresponding to the equilibrium PDF
        and force.

    """
    
    n_traj = len(z_dedrift) # Length of the whole trajectory

    # To keep the Peq and Feq from each noisy trajectories
    # One line = one traj
    # One column = one time
    all_x_pdf_z, all_width_pdf_z, all_y_pdf_z = np.zeros((n_noisy, n_bins)), np.zeros((n_noisy, n_bins)), np.zeros((n_noisy, n_bins))
    all_x_F_eq, all_width_F_eq, all_y_F_eq = np.zeros((n_noisy, n_bins)), np.zeros((n_noisy, n_bins)), np.zeros((n_noisy, n_bins))

    for i in range(n_noisy):
        traj_z = np.copy(z_dedrift) + np.random.normal(0, 5e-9, n_traj)

        x_pdf_z, width_pdf_z, y_pdf_z = logarithmic_hist(traj_z, min_distance, max_distance, num=n_bins+1, base=2) # Here num is the nb of borders.
        x_F_eq, width_F_eq, y_F_eq = logarithmic_hist(traj_z, min_distance, max_distance, num=n_bins+1, base=2)
        y_F_eq = np.gradient(np.log(y_F_eq), x_F_eq) * 4e-21

        all_x_pdf_z[i,:] = x_pdf_z
        all_width_pdf_z[i,:] = width_pdf_z
        all_y_pdf_z[i,:] = y_pdf_z
        all_x_F_eq[i,:] = x_F_eq
        all_width_F_eq[i,:] = width_F_eq
        all_y_F_eq[i,:] = y_F_eq

    x_pdf_z = np.nanmean(all_x_pdf_z, axis=0)
    width_pdf_z = np.nanmean(all_width_pdf_z, axis=0)
    y_pdf_z = np.nanmean(all_y_pdf_z, axis=0)
    std_y_pdf_z = np.nanstd(all_y_pdf_z, axis=0)
    err_y_pdf_z = np.nanstd(all_y_pdf_z, axis=0) #/ np.sqrt(n_noisy) 
    # First guess of the error on the Peq 

    x_F_eq = np.nanmean(all_x_F_eq, axis=0)
    width_F_eq = np.nanmean(all_width_F_eq, axis=0)
    y_F_eq = np.nanmean(all_y_F_eq, axis=0)
    std_y_F_eq = np.nanstd(all_y_F_eq, axis=0)
    err_y_F_eq = np.nanstd(all_y_F_eq, axis=0) #/ np.sqrt(n_noisy) 
    # First guess of the error on the Feq 
    
    res_P_eq = {
        'x_pdf_z': x_pdf_z, 
        'width_pdf_z': width_pdf_z, 
        'y_pdf_z': y_pdf_z, 
        'std_y_pdf_z': std_y_pdf_z, 
        'err_y_pdf_z': err_y_pdf_z, 
    }
    
    res_F_eq = {
        'x_F_eq': x_F_eq, 
        'width_F_eq': width_F_eq, 
        'y_F_eq': y_F_eq, 
        'std_y_F_eq': std_y_F_eq, 
        'err_y_F_eq': err_y_F_eq, 
    }
    
    return res_P_eq, res_F_eq


# =============================================================================
# Moments 
# =============================================================================

def moment(n, x, t, return_std = False, retrieve_mean = False, mean_abs = False):
    """
    Parameters
    ----------
    n: int, moment of order n
    x: array containing the values of interest
    t: an array containing differences of indexes.
    """
    res = np.zeros(len(t))
    std = np.zeros(len(t))
    for i, j in enumerate(t):
        #x_tau = x[j:] - np.mean(x[j:])
        #x_zer = x[0:-j] - np.mean(x[0:-j])
        #res[i] = np.nanmean((x_tau - x_zer)**n)
        Delta_x = x[j:] - x[0:-j]
        if retrieve_mean:
            distribution = (Delta_x - np.mean(Delta_x)) ** n
        else:
            distribution = (Delta_x) ** n
        if mean_abs:
            distribution = np.abs(distribution)
        res[i] = np.nanmean(distribution)
        std[i] = np.nanstd(distribution)
    if return_std:
        return res, std
    else:
        return res

def cumulant4(x, t, return_std = False):
    """
    Compute the fourth-order cumulant, according to its full expression.
    Use the fonction moment and compute the std as "relative std".
    """
    m1, std_m1 = moment(1, x, t, return_std=True)
    m2, std_m2 = moment(2, x, t, return_std=True)
    m3, std_m3 = moment(3, x, t, return_std=True)
    m4, std_m4 = moment(4, x, t, return_std=True)
    # res = (m4 - 4*m1*m3 - 3*(m2**2) + 12*(m1**2)*m2 - 6*(m1**4)) / 24
    res = (m4 - 3*(m2**2))
    std = res * np.sqrt((std_m4/m4)**2 + (2*std_m2/m2)**2)
    if return_std:
        return res, std
    else:
        return res

def velocity(x, di, fps, method='center'):
    v = np.zeros(np.shape(x))
    dt = di / fps
    if method == 'center':
        v[:di] = (x[1:di+1] - x[:di]) / dt
        v[di:-di] = (x[2*di:] - x[0:-2*di]) / (2*dt)
        v[-di:] = (x[-di+1:] - x[-di:-1]) / dt
    # if method == 'Ito':
    #     v[:-di] = (x[di:] - x[:-di]) / dt
    #     v[-di:] = x
    return v


# =============================================================================
# Probability of displacement
# =============================================================================


# =============================================================================
# Diffusion profiles
# =============================================================================

def Compute_diffusion(pos, dt, z_min=None,z_max=None, N=20, ordre=4, method="Vestergaard"):
    """
    Function using the SFI from Ronseray et al. 
    Return the diffusion coefficient profiles, along the the different axis.
    
    Parameters
    ----------
    pos : np.array 
        (len(pos),3) 3 dimensinal trajectory of the particle
    dt : float
        time step in s
    z_min, z_max : floats
        z bounds to get Di from the computed basis using SFI
    N : int
        length of the returned diffusion arrays
        
    Returns
    -------
    Dxm, Dym, Dzm, z : arrays
        Diffusions profiles and corresponding z
    projections_error : float
        Error
    """ 
    
    tlist = np.arange(len(pos)) * dt
    xlist = np.ones((len(pos),1,3))
    xlist[:,0,:] = pos
    data = StochasticTrajectoryData(xlist,tlist)

    S = StochasticForceInference(data)
    S.compute_diffusion(method=method, basis={"type" : "polynomial", "order" : ordre})
    
    S.compute_diffusion_error(maxpoints=N)
    projections_error = S.diffusion_projections_self_consistent_error
    
    dir1 = np.zeros(3); dir1[0] = 1
    dir2 = np.zeros(3); dir2[1] = 1
    dir3 = np.zeros(3); dir3[2] = 1

    Rmin=data.X_ito.min(axis=(0,1))
    Rmax=data.X_ito.max(axis=(0,1))
    
    xbin=np.linspace(Rmin[0],Rmax[0],N)
    ybin=np.linspace(Rmin[1],Rmax[1],N)
    
    if z_min == None:
        z_min = Rmin[2]
    if z_max == None:
        z_max = Rmax[2]
        
    zbin = np.linspace(z_min, z_max, N)
        
    positions = [a * dir1 + b * dir2  + c * dir3 
                 for a in xbin for b in ybin for c in zbin]
    
    NN = len(positions)
    gridX,gridY,gridZ = np.zeros(NN),np.zeros(NN),np.zeros(NN)
    Dx,Dy,Dz = np.zeros(NN),np.zeros(NN),np.zeros(NN)

    
    for n, pos in enumerate(positions):
        
        gridX[n] = dir1.dot(pos)
        gridY[n] = dir2.dot(pos)
        gridZ[n] = dir3.dot(pos)
        
        
        tensor = S.D_ansatz(pos.reshape((1,3)))
        Dx[n], Dy[n], Dz[n] = np.squeeze(tensor.diagonal(axis1 = 2))
        
    
    inflate = lambda a : np.reshape(a,(N,N,N))
    
    to_inflate = Dx, Dy, Dz, gridZ
    Dx, Dy, Dz, zz = map(inflate, to_inflate)
    del to_inflate
    
    to1d = lambda a : np.nanmean(np.nanmean(a, axis = 0),0)
    to_1d = Dx, Dy, Dz, zz
    
    Dxm, Dym, Dzm, z = map(to1d,to_1d)
    
    return  Dxm, Dym, Dzm, z, projections_error


def two_compute_diffusion(pos, fps, z_min, z_max, N_bins, order_perp=3, order_para=3, method="Vestergaard"):
    
    if order_perp == order_para:
        Dx, Dy, Dz, z_D, err_Dx = Compute_diffusion(
            pos, 1/fps, 
            z_min=z_min, z_max=z_max, N=N_bins, 
            ordre=order_perp, 
            method=method, 
        )
        
        err_Dz = err_Dx
    
    else:
        Dx, Dy, _, _, err_Dx = Compute_diffusion(
            pos, 1/fps, 
            z_min=z_min, z_max=z_max, N=N_bins, 
            ordre=order_para, 
            method=method, 
        )
        _, _, Dz, z_D, err_Dz = Compute_diffusion(
            pos, 1/fps, 
            z_min=z_min, z_max=z_max, N=N_bins, 
            ordre=order_perp, 
            method=method, 
        )
        
    del pos
    
    return Dx, Dy, Dz, z_D, err_Dx, err_Dz


# =============================================================================
# Total force
# =============================================================================


def vzz(bins_edges, traj_z, times=[1], fps=100):
    """
    Compute the velocity profile v_z(z) of a confined particle, 
    according to Ito's convention.

    Parameters
    ----------
    bins_edges : array
        The edges of the bins used.
    traj_z : array
        Trajectory (z-coordinate) of the particle.
    times : list, optional
        List of the time steps i to compute velocities: z(t+idt) - z(t). 
        The default is [1].
    fps : int, optional
        Framerate. The default is 100 Hz.

    Returns
    -------
    vz : array
        Velocity profile of the particle, corresponding to bins_edges.
    err_vz : array
        Corresponding error.

    """
    
    # We first get all the different variables of the problem
    Y = bins_edges
    zz = traj_z
    
    # We choose over wich time we want to compute the diffusion coefficient     
    # We initialize the variable used to store the results.
    vz = np.zeros((len(times), len(Y[:-1])))
    errvz = np.zeros((len(times), len(Y[:-1])))
    
    for n, i in enumerate(times):
        # Compute the Delta x = x(Dt + t) - x(t) for given Dt -- same over y
        Dzs = zz[i:] - zz[0:-i]
            
        # Now for each z-bin we are going to measure the diffusion coefficient.
        for m in range(len(Y)-1):
            # We take the Dz corresponding the actual bin 
            dz = Dzs[(zz[:-i] > Y[m]) & (zz[:-i] < Y[m+1])]
            vz[n,m] = np.nanmean(dz) / (i/fps)
            #errvz[n,m] = np.nanstd(dz) / (i/fps)
            errvz[n, m] = len(dz)
    
    vz = np.nanmean(vz, axis=0)
    err_vz = np.nanmean(errvz, axis=0)
     
    return vz, err_vz


def Fz_tot(a, z_dedrift, x_pdf_z, width_pdf_z, z_D, D, order_perp, eta=0.001):
    
    # Interpolation on the diffusion profile
    # The order of the polynomial used should be the same as the order used 
    # in Ronceray's computation of the diffusion (Compute_diffusion).
    # [coef3, coef2, coef1, coef0] 
    coefs = np.polyfit(z_D, D, order_perp) #, w=pdf_z/np.max(pdf_z))
    # p = lambda x: coef3 * x**3 + coef2 * x**2 + coef1 * x + coef0
    p = np.poly1d(coefs)
    # Here, we have the experimental function Dz(z).
    # dpdz = lambda x: 3 * coef3 * x**2 + 2 * coef2 * x + coef1
    dpdz = np.polyder(p, 1)
    # Here, we have the experimental derivative of Dz(z).
    
    # The total force is computed using diffusion profiles and drifts. 
    # The total force must be computed using Ito's convention 
    # (i.e. using the left border of the chosen bins).
    # In the mean time, the total force must be computed for the same set of z 
    # than the equilibrium force (i.e. for the same bins as the pdf). 
    # So the bins centers of the pdf (i.e. the equilibrium force) are 
    # considered as the left of the bins of the total force.
    bins_left = x_pdf_z[:]
    bins_edges = np.concatenate(
        (
            bins_left, np.array([x_pdf_z[-1] + width_pdf_z[-1] / 2]), 
        ) 
    )
    
    Dz_z_exp = p(bins_left)
    Dz_z_prime_exp = dpdz(bins_left)

    vz, err_vz = vzz(bins_edges, z_dedrift, times=np.arange(2,3))

    F1 = 6 * np.pi * eta * 1 / Dz_z_exp * a * vz # Drag term
    F2 = - 4e-21 * Dz_z_prime_exp / Dz_z_exp # Spurious drift term
    res = F1 + F2
    
    return res


# =============================================================================
# Allan variance
# =============================================================================

def _allan_variance(x, m, dt):
    """
    Compute the Allan variance, at one time m*tau.

    Parameters
    ----------
    x : np.ndarray
        Array from which we want the Allan varaince
    m : int
        To choose a multiple of the timestep tau.
    dt : float
        Smallest timestep available.

    Returns
    -------
    res : float
        Allan variance, at one time m*tau.

    """
    # res = (x[2*m:] - 2*x[m:-m] + x[:-2*m]) ** 2
    # res = 1/2 / (m*tau)**2 * np.nanmean(res)
    n = len(x) // m
    res = np.zeros(n)
    tau = m * dt
    t = np.linspace(0, tau, m)
    for i in range(len(res)):
        I1 = x[(i)*m:(i+1)*m]
        I2 = x[(i+1)*m:(i+2)*m]
        integrande = I2 - I1
        res[i] = np.trapz(integrande, t)
    res = np.nanmean(res)
    res = 1/2 / (m*dt)**2 * res
    return res

def allan_variance(x, M, dt):
    """
    Compute the Allan variance, at several times m*tau.

    Parameters
    ----------
    M : list or np.ndarray of int
        Several int to choose several multiples of the timestep tau.

    Returns
    -------
    res : TYPE
        Allan variance, at several times m*tau.

    """
    res = np.zeros(len(M))
    for i, m in enumerate(M):
        res[i] = _allan_variance(x, m, dt)
    return res

def allan_deviation(x, M, dt):
    return np.sqrt(allan_variance(x, M, dt))


# =============================================================================
# TO DO : class
# =============================================================================

# class ComputeObservables(object):
    
#     def __init__(
#             self, 
#             x, y, z, 
#             fps, 
#             r, 
#             ratioBC=None, 
#             zmin_P=50e-9, 
#             zmax_P=1.5e-6, 
#             N_P=40, 
#             zmin_D=None, 
#             zmax_D=None, 
#             N_D=None, 
#             order_perp=3, 
#             order_para=None, 
#         ):
        
#         self.x, self.y, self.z = x, y, z
#         self.fps = fps
    
#     def movmin(self, datas, k):
#         result = np.empty_like(datas)
#         start_pt = 0
#         end_pt = int(np.ceil(k / 2))

#         for i in range(len(datas)):
#             if i < int(np.ceil(k / 2)):
#                 start_pt = 0
#             if i > len(datas) - int(np.ceil(k / 2)):
#                 end_pt = len(datas)
#             result[i] = np.nanmin(datas[start_pt:end_pt])
#             start_pt += 1
#             end_pt += 1

#         return result


