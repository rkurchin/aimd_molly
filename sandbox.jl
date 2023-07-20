using SimpleCrystals
using LinearAlgebra
using DFTK
using Molly
using GLMakie

# convert a `Crystal` object to an AtomsBase system so DFTK can read it in
function FlexibleSystem(c::Crystal)
    particles = [AtomsBase.Atom(a.sym, a.position) for a in c.atoms] # this isn't crucial, but AtomsIO can't write to CIF otherwise
    bb = [bounding_box(c)[1,:], bounding_box(c)[2,:], bounding_box(c)[3,:]]
    FlexibleSystem(particles, bb, boundary_conditions(c))
end

# a few other convenient conversion-related functions
positions(m::Model) = [m.lattice * pos .* u"Å" for pos in m.positions]
frac_coords(vec, lat) = [ustrip(dot(vec, el)/norm(el)^2) for el in lat]

function get_frac_coords(coords, model::Model)
    lat = auconvert.(Ref(u"Å"), model.lattice) # DFTK is in Bohr!
    lat_vecs = [lat[:, 1], lat[:, 2], lat[:, 3]]
    frac_coords.(coords, Ref(lat_vecs))
end

# getting forces from DFTK
function get_forces(coords, dftk_model=dftk_model)
    # calculate fractional coordinates and update positions
    dftk_model = Model(dftk_model, positions=get_frac_coords(coords, dftk_model))

    # build new basis
    basis = PlaneWaveBasis(dftk_model; Ecut, kgrid)

    # and calculate
    scfres = self_consistent_field(basis, tol=1e-5);
    [uconvert.(molly_sys.force_units, f .* u"Eh_au/bohr/mol") for f in compute_forces_cart(scfres)]
end

# define a new Molly interaction using this
struct DFTKInter
    model::Model
end

function Molly.forces(inter::DFTKInter, sys, neighbors=nothing; n_threads=Threads.nthreads())
    return get_forces(sys.coords, inter.model)
end

# build initial geometry
silicon_crystal = Diamond(5.431u"Å", :Si, SVector{3}([1,1,1]))

# set up initial DFTK stuff
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si for i in 1:8]
dftk_model = Model(model_LDA(FlexibleSystem(silicon_crystal)), 
                   atoms = atoms, 
                   temperature = 0.001, 
                   smearing = DFTK.Smearing.FermiDirac(),
                   symmetries = false)
# pos = dftk_model.positions
# pos[1] = pos[1] .+ rand(3) * 1e-3 
# dftk_model = Model(dftk_model, positions=pos)
kgrid = [2, 2, 2]
Ecut = 5

# set up Molly stuff
temp = 10u"K"
velocities = [random_velocity(atom.mass, temp) for atom in silicon_crystal.atoms]

molly_sys = System(c, 
                   general_inters = (DFTKInter(dftk_model),),
                   velocities = velocities,
                   loggers = (coords=CoordinateLogger(10),))

simulator = VelocityVerlet(dt=0.002u"ps",
                        #    coupling=AndersenThermostat(temp, 1.0u"ps")
                           )

simulate!(molly_sys, simulator, 5)
visualize(molly_sys.loggers.coords, molly_sys.boundary, "sim.mp4")
