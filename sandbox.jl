using AtomsBase
using SimpleCrystals
using UnitfulAtomic
using LinearAlgebra
using DFTK
using Molly
using GLMakie

# define a few helpful functions
# positions(m::Model) = [m.lattice * pos .* u"Å" for pos in m.positions]
frac_coords(vec, lat) = [ustrip(dot(vec, el)/norm(el)^2) for el in lat]

function get_frac_coords(coords, model::Model)
    lat = auconvert.(Ref(u"Å"), model.lattice) # DFTK is in Bohr!
    frac_coords.(coords, Ref([lat[:, 1], lat[:, 2], lat[:, 3]]))
end

# getting forces from DFTK
function get_forces(coords, dftk_model; kgrid = [2, 2, 2], Ecut = 5)
    # calculate fractional coordinates and update positions
    new_model = Model(dftk_model, positions=get_frac_coords(coords, dftk_model))

    # build new basis
    basis = PlaneWaveBasis(new_model; Ecut, kgrid)

    # and calculate
    scfres = self_consistent_field(basis, tol=1e-5; callback=x->print()); # blank callback to print less stuff
    compute_forces_cart(scfres) .* u"Eh_au/bohr"
end

# define a new Molly interaction using this
struct DFTKInter
    model::Model
end

Molly.forces(inter::DFTKInter, sys, neighbors=nothing; n_threads=Threads.nthreads()) = get_forces(sys.coords, inter.model)

# build initial geometry
silicon_crystal = Diamond(5.431u"Å", :Si, SVector{3}([1,1,1]))

# set up initial DFTK stuff
atoms = [ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4")) for i in 1:length(silicon_crystal)]
dftk_model = Model(model_LDA(FastSystem(silicon_crystal)), 
                   atoms = atoms, 
                   temperature = 0.001, 
                   smearing = DFTK.Smearing.FermiDirac(), 
                   symmetries = false)

# set up Molly stuff
molly_sys = System(silicon_crystal, 
                   general_inters = (DFTKInter(dftk_model),),
                   velocities = [random_velocity(atom.atomic_mass, 100u"K") for atom in silicon_crystal.atoms],
                   force_units = u"Eh_au/bohr",
                   loggers = (coords=CoordinateLogger(5),
                              velocities=VelocityLogger(5),
                              )
                              )

simulator = VelocityVerlet(dt=0.002u"ps")

# simulate!(molly_sys, simulator, 300)
# visualize(molly_sys.loggers.coords, molly_sys.boundary, "sim_1ps.mp4"; trails=5, markersize=1)
