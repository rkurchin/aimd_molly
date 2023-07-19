using SimpleCrystals
using LinearAlgebra
# using AtomsView
using DFTK
using Molly

# define some helpful things
function FlexibleSystem(c::Crystal)
    particles = [AtomsBase.Atom(a.sym, a.position) for a in c.atoms] # this isn't crucial, but AtomsIO can't write to CIF otherwise
    bb = [bounding_box(c)[1,:], bounding_box(c)[2,:], bounding_box(c)[3,:]]
    FlexibleSystem(particles, bb, boundary_conditions(c))
end

positions(m::Model) = [m.lattice * pos .* u"Å" for pos in m.positions]
frac_coords(vec, lat) = [ustrip(dot(vec, el)/norm(el)^2) for el in lat]

function get_frac_coords(coords, model::Model)
    lat = auconvert.(Ref(u"Å"), model.lattice) # DFTK is in Bohr!
    lat_vecs = [lat[:, 1], lat[:, 2], lat[:, 3]]
    frac_coords.(coords, Ref(lat_vecs))
end

# build initial geometry
silicon_crystal = Diamond(5.431u"Å", :Si, SVector{3}([1,1,1]))

# set up Molly System object
molly_sys = System(c)

# and DFTK stuff
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si for i in 1:8]
dftk_model = Model(model_LDA(FlexibleSystem(silicon_crystal)), atoms=atoms, temperature=0.001, smearing=DFTK.Smearing.FermiDirac())
kgrid = [2, 2, 2]
Ecut = 5

# now just figure out how to pass back and forth...
function get_forces(coords, dftk_model=dftk_model)
    # calculate fractional coordinates and update positions
    dftk_model = Model(dftk_model, positions=get_frac_coords(coords, dftk_model))

    # build new basis
    basis = PlaneWaveBasis(dftk_model; Ecut, kgrid)

    # and calculate
    scfres = self_consistent_field(basis, tol=1e-5);
    compute_forces_cart(scfres)
end

# define new Molly interaction
struct DFTKInter
    model::Model
end

function Molly.forces(inter::DFTKInter, sys, neighbors=nothing; n_threads=Threads.nthreads())
    return get_forces(sys.coords, inter.model)
end