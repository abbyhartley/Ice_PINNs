# -*- coding: utf-8 -*-
"""pinn_iso_bm_sherlock.ipynb

# Isotropic viscosity inversion of ice shelves via PINNs
"""

output_dir = 'figs'
os.makedirs(output_dir, exist_ok=True)

"""# Setting hyperparameters

hyper-parameters used for the training. 

"""

import glob

# select a random seed
seed = 2134
key = random.PRNGKey(seed)
np.random.seed(seed)

# create the subkeys
keys = random.split(key, 4)

# select the size of neural network
n_hl = 6
n_unit = 30
# set the weight for 1. equation loss and 2. boundary condition loss
lw = [0.05, 0.1]

# number of sampling points
n_smp = 5000    # for velocity data
nh_smp = 4500   # for thickness data
n_col = 5000    # for collocation points
n_cbd = 400     # for boundary condition (calving front)
# group all the number of points
n_pt = jnp.array([n_smp, nh_smp, n_col, n_cbd], dtype='int32')
# double the points for L-BFGS training
n_pt2 = n_pt * 2

"""# Data Loading
load and normalize tne observed data before the PINN training
"""

# load .mat data (from real data in diffice-jax)
rawdata = loadmat('data_pinns_Amery.mat')
xd = rawdata['xd']
yd = rawdata['yd']
hd_orig = rawdata['hd']  # original ice thickness, will replace with bedmap

# load bedmap data
folder = os.path.expanduser("bedmap_data")
gpq_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.gpq')]
bedmap_points = gpd.GeoDataFrame(pd.concat([gpd.read_parquet(f) for f in gpq_files], ignore_index=True))
bedmap_points = bedmap_points.dropna(subset=['Mean_thick'])

# extract coordinates and thickness
# ensure CRS is set
if bedmap_points.crs is None:
    bedmap_points.set_crs(epsg=4326, inplace=True)
# reproject to polar stereographic (to match DIFFICE grid)
bedmap_proj = bedmap_points.to_crs(epsg=3031)  # EPSG:3031 = Antarctic Polar Stereographic
x_bedmap = bedmap_proj.geometry.x.values
y_bedmap = bedmap_proj.geometry.y.values
thickness_bedmap = bedmap_proj['Mean_thick'].values

# Current DIFFICE (unshifted) bounds
x_min_dif, x_max_dif = np.nanmin(xd), np.nanmax(xd)
y_min_dif, y_max_dif = np.nanmin(yd), np.nanmax(yd)

# Bedmap bounds (already reprojected to EPSG:3031)
x_min_bed, x_max_bed = np.min(x_bedmap), np.max(x_bedmap)
y_min_bed, y_max_bed = np.min(y_bedmap), np.max(y_bedmap)

# Get valid bounding box of Amery grid
xd_shifted = xd - 2_560_000
yd_shifted = yd - 2_160_000

# Use the shifted bounding box to crop Bedmap points
margin = 10_000  # 10 km extra buffer
x_min = np.nanmin(xd_shifted) - margin
x_max = np.nanmax(xd_shifted) + margin
y_min = np.nanmin(yd_shifted) - margin
y_max = np.nanmax(yd_shifted) + margin

# Crop Bedmap points to that bounding box
in_bounds = ((x_bedmap >= x_min) & (x_bedmap <= x_max) &
    (y_bedmap >= y_min) & (y_bedmap <= y_max))

# Keep only valid thickness values within Amery bounds
valid_bedmap = (thickness_bedmap > 5) & in_bounds
x_bedmap_valid = x_bedmap[valid_bedmap]
y_bedmap_valid = y_bedmap[valid_bedmap]
thickness_valid = thickness_bedmap[valid_bedmap]
points_bedmap = np.column_stack((x_bedmap_valid, y_bedmap_valid))

# Flatten shifted coordinates
x_flat = xd_shifted.ravel()
y_flat = yd_shifted.ravel()
grid_coords = np.column_stack((x_flat, y_flat))

# Downsample Bedmap points for convex hull
step = 8
pb_sub = points_bedmap[::step]
tv_sub = thickness_valid[::step]

# Build convex hull and mask extrapolation
tri = Delaunay(pb_sub)
in_hull = tri.find_simplex(grid_coords) >= 0

# Interpolate only onto shifted grid
interp_vals = griddata(points_bedmap, thickness_valid, grid_coords, method='linear')
interp_vals[~in_hull] = np.nan

# Reshape back to 2D
hd_newgrid = interp_vals.reshape(xd.shape)

# Replace in rawdata
rawdata['hd'] = hd_newgrid

plt.imshow(np.ma.masked_where(np.isnan(hd_newgrid), hd_newgrid), origin='lower', cmap='viridis', vmin=1, vmax=1200)
plt.colorbar()
plt.title("Interpolated Bedmap Thickness Field")
plt.savefig(f"{output_dir}/orig_thick_interp.png", dpi=300)
plt.show()

hd_valid = hd_newgrid[~np.isnan(hd_newgrid)]

print("Bedmap Thickness (hd_newgrid) Summary:")
print(f"  Num valid: {hd_valid.size}")
print(f"  Min: {np.min(hd_valid):.2f}")
print(f"  Max: {np.max(hd_valid):.2f}")
print(f"  Mean: {np.mean(hd_valid):.2f}")
print(f"  Median: {np.median(hd_valid):.2f}")
print(f"  Std dev: {np.std(hd_valid):.2f}")

# update xd_h and yd_h to match valid locations in hd_newgrid
valid_mask = ~np.isnan(hd_newgrid)
xd_h = xd[valid_mask]
yd_h = yd[valid_mask]
# reshape to 2D columns just in case
xd_h = xd_h.reshape(-1, 1)
yd_h = yd_h.reshape(-1, 1)

# flatten hd_newgrid and filter only valid values
hd_flat = hd_newgrid[~np.isnan(hd_newgrid)].reshape(-1, 1)

# update rawdata
# hd, xd_h, yd_h should all be 1D arrays of same length for initialization to work
rawdata['hd'] = hd_flat
rawdata['xd_h'] = xd[~np.isnan(hd_newgrid)].reshape(-1, 1)
rawdata['yd_h'] = yd[~np.isnan(hd_newgrid)].reshape(-1, 1)

# normalize and prep for PINN training
data_all = normdata_pinn(rawdata)
scale = data_all[4][0:2]

"""# Initialization

initialize the neural network and loss function
"""

# initialize the weights and biases of the network
trained_params = init_pinn(keys[0], n_hl, n_unit)

# create the solution function
pred_u = solu_pinn()

# create the data sampling function for Adam training
dataf = dsample_pinn(data_all, n_pt)
keys_adam = random.split(keys[1], 5)
# generate the data
data = dataf(keys_adam[0])

# create the data sampling function for L-BFGS training
dataf_l = dsample_pinn(data_all, n_pt2)
key_lbfgs = random.split(keys[2], 5)

# group the gov. eqn and bdry cond.
eqn_all = (ssa_iso, dbc_iso)
# create the loss function
NN_loss = loss_iso_pinn(pred_u, eqn_all, scale, lw)
# calculate the initial loss and set it as the reference value for loss
NN_loss.lref = NN_loss(trained_params, data)[0]

"""# Network training

Since the real ice shelf data has more complicated profile than the synthetic data. 10000 iterations of Adam, followed by another 10000 iterations of L-BFGS can only infer a **very rough** profile of the ice viscosity.

To train a high-accurate model, the number of iterations required for both Adam and L-BFGS optimization is more than 100k.

Extra training using L-BFGS to reach higher accuracy

Recommended number of iterations: 10000
"""

# Fixed version of save_training_snapshots function

def save_training_snapshots(params, data_all, grid_shape, xd, yd, output_dir, iteration, method_name, scale):
    """Save training snapshots including viscosity and residual plots"""
    try:
        # Create the function group needed for predict_pinn (same as your working code)
        f_u = lambda x: pred_u(params, x)
        f_gu = lambda x: vectgrad(f_u, x)[0][:, 0:6]
        func_all = (f_u, f_gu, ssa_iso)

        # Get current predictions using predict_pinn (same as your working code)
        results = predict_pinn(func_all, data_all)

        # Save viscosity snapshot (using 'mu' key like in your working code)
        try:
            viscosity = results['mu']
            viscosity_reshaped = viscosity.reshape(grid_shape)

            # Handle NaN/Inf by computing color limits from finite values only
            finite_mask = np.isfinite(viscosity_reshaped)
            if finite_mask.any():
                vmin = np.min(viscosity_reshaped[finite_mask])
                vmax = np.max(viscosity_reshaped[finite_mask])
            else:
                print(f"Warning: All viscosity values are NaN/Inf at iteration {iteration}")
                vmin, vmax = 0, 1  # fallback values

            fig, ax = plt.subplots(figsize=(10, 5), dpi=70)
            h = ax.imshow(viscosity_reshaped, interpolation='nearest', cmap='rainbow',
                          extent=[0., 50000., 0,  80000.],
                          origin='lower', aspect='auto',
                          vmin=vmin, vmax=vmax)  # Use computed finite limits
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(h, cax=cax)
            ax.set_xlabel('$x$', fontsize=15)
            ax.set_ylabel('$y\ $', fontsize=15, rotation=0)
            ax.set_title(f'Inferred viscosity $\mu(x,y)$ - {method_name} Iteration {iteration}', fontsize=15)
            fig.savefig(f"{output_dir}/mu_e1_e2_snaps/viscosity_{method_name}_iter_{iteration}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved viscosity at iteration {iteration}")
        except Exception as e:
            print(f"Failed to save viscosity at iteration {iteration}: {e}")

        # Save residual snapshots with FIXED COLOR LIMITS
        residual_keys = ['e1', 'e2']
        for key in residual_keys:
            if key in results:
                try:
                    residual = results[key]
                    reshaped = residual.reshape(grid_shape)

                    # Check if xd and yd have valid extents
                    if not (np.isfinite(xd.min()) and np.isfinite(xd.max()) and
                            np.isfinite(yd.min()) and np.isfinite(yd.max())):
                        print(f"Warning: xd/yd contains NaN/Inf at iteration {iteration}")
                        extent = [0, 1, 0, 1]  # fallback extent
                    else:
                        extent = [xd.min(), xd.max(), yd.min(), yd.max()]

                    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
                    # Use FIXED color limits to avoid NaN/Inf issues completely
                    h = ax.imshow(reshaped, interpolation='nearest', cmap='RdBu_r',
                                  extent=extent,
                                  origin='lower', aspect='auto',
                                  vmin=-1000, vmax=1000)  # FIXED LIMITS
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="4%", pad=0.05)
                    fig.colorbar(h, cax=cax)
                    ax.set_title(f'{key} Residual - {method_name} Iteration {iteration}', fontsize=14)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    fig.savefig(f"{output_dir}/mu_e1_e2_snaps/{key}_residual_{method_name}_iter_{iteration}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved {key} residual at iteration {iteration}")
                except Exception as e:
                    print(f"Failed to save {key} residual at iteration {iteration}: {e}")

        print(f"Saved snapshots for {method_name} iteration {iteration}")

    except Exception as e:
        print(f"Error saving snapshots at iteration {iteration}: {e}")
        import traceback
        traceback.print_exc()  # This will show exactly where the error occurs

def create_animation_gifs(output_dir):
    """
    Create GIF animations from saved frames
    """
    animation_dir = f"{output_dir}/animation_frames"

    # Create viscosity animation
    viscosity_files = sorted(glob.glob(f"{animation_dir}/viscosity_*.png"))
    if viscosity_files:
        images = []
        for filename in viscosity_files:
            images.append(imageio.imread(filename))
        imageio.mimsave(f"{output_dir}/viscosity_evolution.gif", images, duration=0.5)
        print(f"Created viscosity animation with {len(images)} frames")

    # Create residual animations
    for residual in ['e1', 'e2']:
        residual_files = sorted(glob.glob(f"{animation_dir}/{residual}_*.png"))
        if residual_files:
            images = []
            for filename in residual_files:
                images.append(imageio.imread(filename))
            imageio.mimsave(f"{output_dir}/{residual}_residual_evolution.gif", images, duration=0.5)
            print(f"Created {residual} residual animation with {len(images)} frames")

# Create directory for animation frames
os.makedirs(f"{output_dir}/animation_frames", exist_ok=True)

# modified adam training

# Define grid shape from your existing xd, yd arrays
grid_shape = xd.shape

# set the learning rate for Adam
lr = 1e-3
# set the training iteration
epoch1 = 10000
snapshot_interval = 100  # Save every 100 iterations

print("Starting Adam training with snapshots...")

# Initialize for Adam training

# Use existing adam_opt but with snapshot functionality
# We'll modify the training to save snapshots during the existing adam_opt call

# Save initial snapshot
save_training_snapshots(trained_params, data_all, grid_shape, xd, yd, output_dir, 0, "Adam", scale)

# Create a custom callback version that saves snapshots
# Since adam_opt doesn't support callbacks, we'll use a wrapper approach
def adam_with_snapshots(key, loss_fn, init_params, data_fn, epochs, lr):
    # Use smaller chunks and save periodically
    chunk_size = snapshot_interval
    current_params = init_params
    all_losses = []

    total_iterations = 0

    for chunk_start in range(0, epochs, chunk_size):
        chunk_epochs = min(chunk_size, epochs - chunk_start)

        # Train for this chunk
        current_params, chunk_losses = adam_opt(key, loss_fn, current_params, data_fn, chunk_epochs, lr=lr)
        all_losses.extend(chunk_losses)

        total_iterations += chunk_epochs

        # Save snapshot after each chunk
        save_training_snapshots(current_params, data_all, grid_shape, xd, yd,
                              output_dir, total_iterations, "Adam", scale)

        print(f"Adam progress: {total_iterations}/{epochs} iterations completed")

        # Generate new key for next chunk
        key, _ = random.split(key)

    return current_params, all_losses

# Run Adam training with snapshots
trained_params, loss1 = adam_with_snapshots(keys_adam[1], NN_loss, trained_params, dataf, epoch1, lr)

# Modified L-BFGS training

# set the training iteration
epoch2 = 10000
# re-sample the data and collocation points
data_l = dataf_l(key_lbfgs[1])

print("Starting L-BFGS training with snapshots...")

# Save initial L-BFGS snapshot
save_training_snapshots(trained_params, data_all, grid_shape, xd, yd, output_dir, 0, "LBFGS", scale)

# Use chunked approach for L-BFGS as well
def lbfgs_with_snapshots(loss_fn, init_params, data, epochs):
    chunk_size = snapshot_interval
    current_params = init_params
    all_losses = []

    total_iterations = 0

    for chunk_start in range(0, epochs, chunk_size):
        chunk_epochs = min(chunk_size, epochs - chunk_start)

        # Train for this chunk
        current_params, chunk_losses = lbfgs_opt(loss_fn, current_params, data, chunk_epochs)
        all_losses.extend(chunk_losses)

        total_iterations += chunk_epochs

        # Save snapshot after each chunk
        save_training_snapshots(current_params, data_all, grid_shape, xd, yd,
                              output_dir, total_iterations, "LBFGS", scale)

        print(f"L-BFGS progress: {total_iterations}/{epochs} iterations completed")

    return current_params, all_losses

# Run L-BFGS training with snapshots
trained_params2, loss2 = lbfgs_with_snapshots(NN_loss, trained_params, data_l, epoch2)

# Create animations after training is complete
print("Creating animation GIFs...")
create_animation_gifs(output_dir/mu_e1_e2_snaps)

print("Training complete! Check the output directory for:")
print(f"- Individual frames in: {output_dir}/animation_frames/")
print(f"- Viscosity evolution GIF: {output_dir}/viscosity_evolution.gif")
print(f"- Residual evolution GIFs: {output_dir}/e1_residual_evolution.gif and {output_dir}/e2_residual_evolution.gif")

"""# Prediction

Compute the solution variables and equation residue at high-resolution grids
"""

# Function of solution and equation residues based on trained networks
f_u = lambda x: pred_u(trained_params2, x)
f_gu = lambda x: vectgrad(f_u, x)[0][:, 0:6]

# group all the function
func_all = (f_u, f_gu, ssa_iso)

# calculate the solution and equation residue at given grids for visualization
results = predict_pinn(func_all, data_all)

"""# Plotting the results:

Compare the synthetic data for either velocity or thickness with the corresponding network approximation
"""

u_g = results['u_g']
u = results['u']

xd = np.nan_to_num(xd, nan=0.0, posinf=0.0, neginf=0.0) # filter out nans/inf values
yd = np.nan_to_num(yd, nan=0.0, posinf=0.0, neginf=0.0)
# Get the shape of the grid
grid_shape = xd.shape  # or yd.shape, should be the same

# Reshape velocity data for plotting
u_g_reshaped = u_g.reshape(grid_shape)
u_reshaped = u.reshape(grid_shape)

fig = plt.figure(figsize = [10, 10], dpi = 70)

ax = plt.subplot(2, 1, 1)
h = ax.imshow(u_g_reshaped, interpolation='nearest', cmap='rainbow',
              extent=[xd.min(), xd.max(), yd.min(), yd.max()],
              origin='lower', aspect='auto', vmin=0, vmax=4e-5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$y$', fontsize=15, rotation=0)
ax.set_title('Observed $u_g(x,y)$ (m/s)', fontsize=15)

ax2 = plt.subplot(2, 1, 2)
h2 = ax2.imshow(u_reshaped, interpolation='nearest', cmap='rainbow',
                extent=[xd.min(), xd.max(), yd.min(), yd.max()],
                origin='lower', aspect='auto', vmin=0, vmax=4e-5)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h2, cax=cax)

ax2.set_xlabel('$x$', fontsize=15)
ax2.set_ylabel('$y$', fontsize=15, rotation=0)
ax2.set_title('PINN Inferred $u(x,y)$ (m/s)', fontsize=15)
fig.savefig(f"{output_dir}/u_comp.png", dpi=300, bbox_inches='tight')

v_g = results['v_g']
v = results['v']

xd = np.nan_to_num(xd, nan=0.0, posinf=0.0, neginf=0.0) # filter out nans/inf values
yd = np.nan_to_num(yd, nan=0.0, posinf=0.0, neginf=0.0)
# Get the shape of the grid
grid_shape = xd.shape  # or yd.shape, should be the same

# Reshape velocity data for plotting
v_g_reshaped = v_g.reshape(grid_shape)
v_reshaped = v.reshape(grid_shape)

fig = plt.figure(figsize = [10, 10], dpi = 70)

ax = plt.subplot(2, 1, 1)
h = ax.imshow(v_g_reshaped, interpolation='nearest', cmap='rainbow',
              extent=[xd.min(), xd.max(), yd.min(), yd.max()],
              origin='lower', aspect='auto', vmin=0, vmax=0.5e-5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$y$', fontsize=15, rotation=0)
ax.set_title('Observed $v_g(x,y)$ (m/s)', fontsize=15)

ax2 = plt.subplot(2, 1, 2)
h2 = ax2.imshow(v_reshaped, interpolation='nearest', cmap='rainbow',
                extent=[xd.min(), xd.max(), yd.min(), yd.max()],
                origin='lower', aspect='auto', vmin=0, vmax=0.5e-5)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h2, cax=cax)

ax2.set_xlabel('$x$', fontsize=15)
ax2.set_ylabel('$y$', fontsize=15, rotation=0)
ax2.set_title('PINN Inferred $v(x,y)$ (m/s)', fontsize=15)
fig.savefig(f"{output_dir}/v_comp.png", dpi=300, bbox_inches='tight')

results.keys()

# plot residuals
residual_keys = ['e1', 'e2', 'e12', 'e21', 'e11', 'e22', 'e13', 'e23']
for key in residual_keys:
    residual = results[key]
    try:
        reshaped = residual.reshape(grid_shape)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        h = ax.imshow(reshaped, interpolation='nearest', cmap='RdBu_r',
                      extent=[xd.min(), xd.max(), yd.min(), yd.max()],
                      origin='lower', aspect='auto',
                      vmin=-1000, vmax=1000)  # fixed color scale
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(h, cax=cax)
        ax.set_title(f'{key}(x,y)', fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.savefig(f"{output_dir}/{key}_residual.png", dpi=300, bbox_inches='tight')
        print(f"Saved {key}_residual_2.png")

        if key == 'e1':
            plt.show()

        plt.close(fig)
    except Exception as e:
        print(f"Failed to plot {key}: {e}")

"""Showing the inferred viscosity via PINNs for the ice shelf"""

# load the PINN inference of viscosity
mu = results['mu']
mu_reshaped = mu.reshape(grid_shape)

fig = plt.figure(figsize = [10, 5], dpi = 70)

ax = plt.subplot(1,1,1)
h = ax.imshow(mu_reshaped, interpolation='nearest', cmap='rainbow',
              extent=[0., 50000., 0,  80000.],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_xlabel('$x$', fontsize = 15)
ax.set_ylabel('$y\ $', fontsize = 15, rotation = 0)
ax.set_title('Inferred viscosity $\mu(x,y)$', fontsize = 15)
fig.savefig(f"{output_dir}/inferred_visc_2.png", dpi=300, bbox_inches='tight')

# log-scaled colorbar for vicosity plot
# avoid log(0): set a small floor value
mu_reshaped = np.where(mu_reshaped <= 0, 1e-3, mu_reshaped)

fig = plt.figure(figsize=[10, 5], dpi=70)

ax = plt.subplot(1, 1, 1)

# Use LogNorm for color scaling
h = ax.imshow(mu_reshaped, interpolation='nearest', cmap='plasma',
              norm=LogNorm(vmin=np.nanmin(mu_reshaped), vmax=np.nanmax(mu_reshaped)),
              extent=[0., 50000., 0, 80000.],
              origin='lower', aspect='auto')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$y$', fontsize=15, rotation=0)
ax.set_title('Inferred viscosity $\\mu(x,y)$ (log scale)', fontsize=15)

fig.savefig(f"{output_dir}/inferred_visc_logscale_2.png", dpi=300, bbox_inches='tight')
plt.show()

# thickness comparison plot
# Flatten full grid for interpolation
grid_points = np.column_stack((xd.ravel(), yd.ravel()))

# Interpolate observed (radar) thickness data
h_g = results['h_g'].ravel()
x_g = results['x_h'].ravel()  # assumes same coords as h2
y_g = results['y_h'].ravel()
points_g = np.column_stack((x_g, y_g))
h_g_grid_flat = griddata(points_g, h_g, grid_points, method='linear')
h_g_grid = h_g_grid_flat.reshape(xd.shape)

# Interpolate inferred PINN thickness
h2 = results['h2'].ravel()
h_x = results['x_h'].ravel()
h_y = results['y_h'].ravel()
points_h = np.column_stack((h_x, h_y))
h2_grid_flat = griddata(points_h, h2, grid_points, method='linear')
h2_grid = h2_grid_flat.reshape(xd.shape)

# Plot both maps
fig = plt.figure(figsize=[10, 10], dpi=70)

# Top: Observed Bedmap thickness (radar)
ax1 = plt.subplot(2, 1, 1)
h1 = ax1.imshow(h_g_grid, interpolation='nearest', cmap='viridis',
                extent=[xd.min(), xd.max(), yd.min(), yd.max()],
                origin='lower', aspect='auto', vmin=150, vmax=1050)
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h1, cax=cax1)

ax1.set_xlabel('$x$', fontsize=15)
ax1.set_ylabel('$y$', fontsize=15, rotation=0)
ax1.set_title('Observed Thickness $h_g(x,y)$ (m)', fontsize=15)

# Bottom: Inferred by PINN
ax2 = plt.subplot(2, 1, 2)
h2_plot = ax2.imshow(h2_grid, interpolation='nearest', cmap='viridis',
                     extent=[xd.min(), xd.max(), yd.min(), yd.max()],
                     origin='lower', aspect='auto', vmin=150, vmax=1050)
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="4%", pad=0.05)
fig.colorbar(h2_plot, cax=cax2)

ax2.set_xlabel('$x$', fontsize=15)
ax2.set_ylabel('$y$', fontsize=15, rotation=0)
ax2.set_title('PINN Inferred Thickness $h(x,y)$ (m)', fontsize=15)

fig.tight_layout()
fig.savefig(f"{output_dir}/thickness_comp_2.png", dpi=300, bbox_inches='tight')
plt.show()