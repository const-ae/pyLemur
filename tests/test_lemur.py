from io import StringIO

import anndata as ad
import formulaic
import numpy as np
import pandas as pd

import pylemur.tl._grassmann
import pylemur.tl.alignment
import pylemur.tl.lemur


def test_design_specification_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat)
    assert model.design_matrix.equals(formulaic.model_matrix("~ condition", dat))

    adata = ad.AnnData(Y)
    model = pylemur.tl.LEMUR(adata, design="~ condition", obs_data=dat)
    assert model.design_matrix.equals(formulaic.model_matrix("~ condition", dat))

    adata = ad.AnnData(Y, obs=dat)
    model = pylemur.tl.LEMUR(adata, design="~ condition")
    assert model.design_matrix.equals(formulaic.model_matrix("~ condition", dat))


def test_numpy_design_matrix_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    design_mat = formulaic.model_matrix("~ condition", dat).to_numpy()
    grouping = np.random.choice(2, size=500)

    ref_model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat).fit().align_with_grouping(grouping)

    model = pylemur.tl.LEMUR(Y, design=design_mat).fit().align_with_grouping(grouping)
    assert np.allclose(model.coefficients, ref_model.coefficients)

    model = pylemur.tl.LEMUR(ad.AnnData(Y), design=design_mat).fit().align_with_grouping(grouping)
    assert np.allclose(model.coefficients, ref_model.coefficients)

    design_df = pd.DataFrame(design_mat, columns=["Intercept", "Covar1"])
    model = pylemur.tl.LEMUR(Y, design=design_df).fit().align_with_grouping(grouping)
    assert np.allclose(model.coefficients, ref_model.coefficients)
    assert np.allclose(model.alignment_coefficients, ref_model.alignment_coefficients)


def test_pandas_design_matrix_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    design_mat = formulaic.model_matrix("~ condition", dat)

    ref_model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat).fit()

    model = pylemur.tl.LEMUR(Y, design=design_mat).fit()
    assert np.allclose(model.coefficients, ref_model.coefficients)

    model = pylemur.tl.LEMUR(ad.AnnData(Y), design=design_mat).fit()
    assert np.allclose(model.coefficients, ref_model.coefficients)


def test_copy_works():
    Y = np.random.randn(500, 30)
    dat = pd.DataFrame({"condition": np.random.choice(["trt", "ctrl"], size=500)})
    model = pylemur.tl.LEMUR(Y, design="~ condition", obs_data=dat)
    cp = model.copy(copy_adata=False)
    cp2 = model.copy(copy_adata=True)
    assert id(model.adata) == id(cp.adata)
    assert id(model.adata) != id(cp2.adata)
    _assert_lemur_model_equal(model, cp)
    _assert_lemur_model_equal(model, cp2, adata_id_equal=False)


def test_predict():
    ## Make sure I get the same results as in R
    # save_func <- function(obj){
    #   readr::format_csv(as.data.frame(obj), col_names = FALSE)
    # }
    # randn <- function(n, m, ...){
    #   matrix(rnorm(n * m, ...), nrow = n, ncol = m)
    # }
    #
    # set.seed(1)
    # Y <- round(randn(5, 30), 2)
    # design <- cbind(1, rep(1:2, each = 15))
    # fit <- lemur(Y, design = design, n_embedding = 3, test_fraction = 0)
    # save_func(Y)
    # save_func(design)
    # save_func(predict(fit))
    Y = np.genfromtxt(
        StringIO(
            "-0.63,-0.82,1.51,-0.04,0.92,-0.06,1.36,-0.41,-0.16,-0.71,0.4,1.98,2.4,0.19,0.48,0.29,-0.57,0.33,-0.54,0.56,-0.62,1.77,-0.64,-0.39,-0.51,0.71,0.06,-1.54,-1.91,-0.75\n0.18,0.49,0.39,-0.02,0.78,-0.16,-0.1,-0.39,-0.25,0.36,-0.61,-0.37,-0.04,-1.8,-0.71,-0.44,-0.14,1.06,1.21,-1.28,0.04,0.72,-0.46,-0.32,1.34,-0.07,-0.59,-0.3,1.18,2.09\n-0.84,0.74,-0.62,0.94,0.07,-1.47,0.39,-0.06,0.7,0.77,0.34,-1.04,0.69,1.47,0.61,0,1.18,-0.3,1.16,-0.57,-0.91,0.91,1.43,-0.28,-0.21,-0.04,0.53,-0.53,-1.66,0.02\n1.6,0.58,-2.21,0.82,-1.99,-0.48,-0.05,1.1,0.56,-0.11,-1.13,0.57,0.03,0.15,-0.93,0.07,-1.52,0.37,0.7,-1.22,0.16,0.38,-0.65,0.49,-0.18,-0.68,-1.52,-0.65,-0.46,-1.29\n0.33,-0.31,1.12,0.59,0.62,0.42,-1.38,0.76,-0.69,0.88,1.43,-0.14,-0.74,2.17,-1.25,-0.59,0.59,0.27,1.59,-0.47,-0.65,1.68,-0.21,-0.18,-0.1,-0.32,0.31,-0.06,-1.12,-1.64\n"
        ),
        delimiter=",",
    ).T
    design = np.genfromtxt(
        StringIO(
            "1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n"
        ),
        delimiter=",",
    )
    results_init = np.genfromtxt(
        StringIO(
            "-0.8554888214077533,-0.2720977014782501,1.467533504177673,-0.22767956411506196,1.2785521085014837,0.14379806862053318,1.4750333827944937,-0.6570301309706671,0.2066831996074998,-0.353750513692208,0.40174695240184183,1.0849159925675056,1.8230740377826868,-0.3796132967907797,1.27432278200101,0.25734634440017745,-0.48925340880447665,0.15787257755416356,0.04938969573830959,0.3192312181063839,-0.3542994869485033,1.401448533272853,-0.3938541882847783,0.08741246846567274,-0.6510475608713524,0.22923361101684514,-0.06804635141148654,-0.8510364064773515,-1.6943910912740754,-1.7500059544823854\n0.22583899533086937,-0.23735468442585927,0.19733713734339808,-0.4421331667319863,0.03586724830483476,0.4198858825154666,-0.08304003977298911,-0.16513486823394496,-0.30042983957286806,-0.3460343445737336,-0.3558867742284205,0.3011982462662531,-0.20850076449582816,-1.0161941132088552,-0.27541891451634165,-0.38785291893731433,-0.12147153601775817,0.9332086847171666,1.6012153818454613,-1.4764027185416573,0.23315085154197657,0.4422452475355182,-0.20026735579063953,0.04360380473394784,1.2311912129895743,-0.3862511983506538,-0.7251300187432477,0.11080434946587961,1.2741080839827195,1.467848129569024\n-0.7940511553704984,0.36872650465867485,-0.696190082245086,0.7842394355421791,-0.2859846254450281,-1.2496520777731421,0.3835639668947345,0.06394998322061843,0.6354423384795286,0.43034530046034825,0.446899273641681,-0.6513965946225246,0.6871089757451906,1.8676308579518373,0.6993678988614832,-0.35853838671959254,1.3529850324988306,-0.246355558204097,1.1840747073603424,-0.40278806288570534,-0.980382841146775,1.0552239017849732,0.9738387917168745,-0.4865414099512409,-0.14227709749759898,-0.07349599449524213,0.7673735611128623,-0.27683325130457614,-1.4140680282113947,-0.2222153640576589\n1.418455323621421,1.003633642928876,-2.249906497292914,0.6558257816955055,-1.7203847145600766,-0.29827588824455953,0.043760865569417634,0.9060713852440867,0.8559072353487625,0.15882996592743862,-1.1213675611187175,-0.13675398777029954,-0.4426075110785659,-0.28963584470503034,-0.27355219556533633,-0.22900366687718224,-1.4215471940245057,0.5395008912619365,0.30620330206252744,-0.9001187449182614,-0.09048745003886999,0.7701978486901692,-1.2331890579233868,-0.0319982155355501,-0.019760633637726566,-0.37122588017711394,-1.2165536309067462,-0.9082217868619842,-0.3910853125055948,-0.8027104686077078\n0.16860548847823306,0.019631704090379926,1.069169569969778,0.4089409239787705,0.8084926808056435,0.6289431224230143,-1.2935687373997728,0.6009167939510944,-0.42509727026387234,1.0706646727858744,1.4570831601240841,-0.7309939926404594,-1.1819487005772338,1.8301112490396232,-0.6209506647651512,-0.1631607424281398,0.3494484690478879,0.3004091825038984,1.248612413250847,-0.5327275335431212,-0.7111287487922291,1.7104365790742557,0.179842226080069,-0.19838902997632976,-0.10222070429937023,-0.02570345245587259,0.10711511924305978,-0.7160399504157087,-1.5156808495142373,-0.8308129777750086\n"
        ),
        delimiter=",",
    ).T

    model = pylemur.tl.LEMUR(Y, design=design, n_embedding=3)
    model.fit(verbose=False)
    pred_init = model.predict()
    assert np.allclose(pred_init, results_init)

    ## Here I have to override all coefficients to make sure that the results match
    # set.seed(1)
    # wide_alignment_coef <- round(cbind(randn(3,4), randn(3,4)), 2)
    # fit@metadata$alignment_coefficients <- array(wide_alignment_coef, dim = c(3,4,2))
    # res <- predict(fit)
    # save_func(fit$base_point)
    # save_func(fit$embedding)
    # save_func(wide_alignment_coef)
    # save_func(cbind(fit$coefficients[,,1], fit$coefficients[,,2]))
    # save_func(res)
    basepoint = np.genfromtxt(
        StringIO(
            "-0.38925206703492543,0.5213219572509142,-0.7231411670117558\n-0.014297729446464128,-0.3059067922758592,0.056315487696737755\n0.2455091078121989,0.48054559390052237,0.133975572527343\n0.7927765077921876,-0.20187676099522883,-0.5743821409539152\n0.39938589098238725,0.602466726782849,0.3550086203707195\n"
        ),
        delimiter=",",
    ).T
    assert np.allclose(pylemur.tl._grassmann.grassmann_angle_from_point(basepoint.T, model.base_point.T), 0)
    emb = np.genfromtxt(
        StringIO(
            "1.307607290530331,1.204126637770359,-2.334941421246343,1.202469151007156,-1.7184261449703435,-0.5734035327253721,-0.6805385978545121,1.3185253662996623,0.8741076652495349,0.8858962317444363,-0.34911335825714623,-0.9635203043565856,-1.052446567304073,1.396858005348387,-0.5172004212354832,-0.5028122904697048,-0.3322559893690832,1.050654556399191,2.1375501746096512,-1.7864259568934735,-0.38553520332697033,1.7579862599667835,-0.44888734172910766,-0.14472710389159946,0.8047342904549903,-0.4509018220855433,-0.8826656042763416,-0.7455036156569727,-0.28188072558002397,0.2106703718482004\n-1.8164798196642507,-0.477295729296259,0.4790240845360194,0.1574495019414363,0.5418966595889152,-1.218733839936566,-0.030245840455864665,-0.6590134762699248,-0.17773495609983386,0.16681479907923033,0.9577303872796369,-0.8649693035413422,0.5516152492211109,1.872319735730998,0.5176225478866773,0.1933219672043183,1.1600064890134691,-0.2055279125365953,0.8140862711968238,0.5522995484090838,-1.0056198219223997,2.0135021529189063,0.9037871448321165,-0.1835087018953936,-0.7687706326544156,0.4350873523204601,1.1121743457677276,-0.6639780098588698,-2.6747181225665626,-1.6821420702286731\n0.18948314499743618,-0.35557961492948087,1.2558867706759906,-0.06200523822352812,0.8150276978150715,0.9017861009351356,-1.7593369587152685,0.38440670001846194,-0.9259345355859308,0.7952852407963301,1.2519868131937648,-0.7881479698006301,-1.7254602945657311,1.1468879075695446,-1.1242857641811612,-0.777687817264553,1.2908968434310275,-0.7487113052869242,0.3633562246175412,-0.7434161046620548,-0.6564971373410212,-1.062751302635433,0.9113503972724165,-0.742431234492101,0.1738627261607038,-0.5327010359755776,0.46611269007910494,0.42937368327065845,0.34776880301511215,1.281474569811115\n"
        ),
        delimiter=",",
    ).T
    wide_alignment_coef = np.genfromtxt(
        StringIO(
            "0,-0.63,1.6,0.49,0,-0.31,-0.62,-0.04\n0,0.18,0.33,0.74,0,1.51,-2.21,-0.02\n0,-0.84,-0.82,0.58,0,0.39,1.12,0.94\n"
        ),
        delimiter=",",
    )
    alignment_coef = np.stack([wide_alignment_coef[:, 0:4], wide_alignment_coef[:, 4:8]], axis=2)
    coef = np.genfromtxt(
        StringIO(
            "-0.39126157680942947,-0.13893603024573095,0.22217190036192688,0.2997192813183081,0.045415132726990666,-0.11388588356030183\n-1.4118852084954074,-0.12010846331372559,-0.042620766619525983,1.1685923275506187,-0.009795610498605375,0.1481261731296855\n0.4301922175223224,0.8308632945684709,-1.7460499853532059,-0.17472584078151635,-0.3588445576311046,1.1196336953167267\n0.010836267191959998,-0.020245706071854806,0.04720615470254291,-0.013801681502086402,0.009718078909566252,-0.032178386499295816\n-0.7178345111625161,-0.6102684315610786,1.1946309212817439,0.46875216448007323,0.2452095180077213,-0.7300769880872995\n"
        ),
        delimiter=",",
    )
    coef = np.stack([coef[:, 0:3], coef[:, 3:6]], axis=2)
    coef = np.einsum("ijk->jik", coef)
    results = np.genfromtxt(
        StringIO(
            "0.9835142473010876,0.5282279297874248,0.3734623735376733,0.29148494672406977,0.3233549213466831,0.8738371099709996,0.5407962262440337,0.5618064038652619,0.4615777825112788,0.28201578105225994,0.06759705564456442,0.8278171090396453,0.3564197483842603,-0.37494408404208857,0.3130324486328697,4.43109412669912,4.485260023784498,-13.556871509080235,-25.740741431276582,19.3507385539798,4.267028891777953,-24.31631072038289,5.55415174991481,0.5751412151679098,-8.850525840651931,3.9506221824433108,9.87464782407815,9.75898404222322,5.923544047383523,0.5432368439394216\n-0.5884851968278264,-0.6980003631236811,1.0095730179286158,-0.706603653513592,0.6775999178228775,0.2960753981255735,-0.04883524121983143,-0.6502161106433406,-0.6374945628932893,-0.46524145497379454,0.0868361701396683,0.24887091370316872,0.08165271296897453,-0.7703009490648383,-0.08543059842869558,1.8195448028133048,2.2357171298099003,-4.317401830679442,-8.088317047084285,6.921096710379453,1.5046074415988786,-7.462954899818863,2.5044021915523067,0.44085644478623487,-2.7813085924525587,1.7337832588534812,3.965667679124676,3.523130209648968,1.7727989396531905,0.2683775618147773\n1.2241353640082717,1.439811782612444,-2.4939338263214896,1.413000126505777,-1.732752410802792,-0.8109483715136666,0.060386636787661846,1.297799915381223,1.325781352884555,0.8252158435825825,-0.46440880845174237,-0.6215041984220009,-0.2577507799745221,1.405770288579147,0.07939708514456326,3.63765958748972,4.311718634450114,-10.958511305614007,-20.4685709755674,15.761881934367485,3.4712598990276082,-19.643794665294983,5.058176520273988,0.500553412576863,-6.917971274367058,3.329924337359561,8.448268016100178,8.2448015916862,5.000276047604282,0.9543282399075053\n-0.7921997533676974,-0.22060767244684792,-0.13854023872636448,-0.07315533443899382,-0.04074837994335717,-0.6868254379104765,0.30181730830558207,-0.4486065557871184,0.016532124694023168,-0.2487768721959298,-0.043942589618246174,-0.18230527640412342,0.5013051052792722,0.22675447837288173,0.33929909418738713,3.3366520193532914,3.389761379749944,-10.811207239740792,-20.286351865125106,15.07164187230232,3.0442924834976672,-18.933270905633723,4.214979326150896,0.264093656883327,-7.2491426213752,2.978123336697904,7.6524353015805255,7.331256553093012,4.067336259849709,-0.0705995572837037\n0.429231215968942,0.4526675323865768,-0.0583697984419273,0.6074892682578291,0.02405662404573697,0.13577913690270105,-0.3608880085761539,0.6353597043295498,0.26375995622833437,0.7194765847659087,0.5782845037350453,-0.326907098443994,-0.38119677146460307,1.1810728627919298,-0.08981571248586934,5.740966466337848,6.297668960348173,-17.056604679816925,-32.1536933270833,24.663672997961317,5.422318375098499,-30.48048219956338,7.551698596077222,0.8269489489661501,-10.99498861966501,5.205042841546413,12.959282592098615,12.608602675876007,7.517277260484411,0.9922891113340572\n"
        ),
        delimiter=",",
    ).T

    model.alignment_coefficients = alignment_coef
    model.base_point = basepoint
    model.embedding = emb
    model.coefficients = coef
    pred_new = model.predict()
    assert np.allclose(pred_new, results)


def test_align_with_grouping():
    ## Make sure I get the same results as in R
    ## Same code as above
    Y = np.genfromtxt(
        StringIO(
            "-0.63,-0.82,1.51,-0.04,0.92,-0.06,1.36,-0.41,-0.16,-0.71,0.4,1.98,2.4,0.19,0.48,0.29,-0.57,0.33,-0.54,0.56,-0.62,1.77,-0.64,-0.39,-0.51,0.71,0.06,-1.54,-1.91,-0.75\n0.18,0.49,0.39,-0.02,0.78,-0.16,-0.1,-0.39,-0.25,0.36,-0.61,-0.37,-0.04,-1.8,-0.71,-0.44,-0.14,1.06,1.21,-1.28,0.04,0.72,-0.46,-0.32,1.34,-0.07,-0.59,-0.3,1.18,2.09\n-0.84,0.74,-0.62,0.94,0.07,-1.47,0.39,-0.06,0.7,0.77,0.34,-1.04,0.69,1.47,0.61,0,1.18,-0.3,1.16,-0.57,-0.91,0.91,1.43,-0.28,-0.21,-0.04,0.53,-0.53,-1.66,0.02\n1.6,0.58,-2.21,0.82,-1.99,-0.48,-0.05,1.1,0.56,-0.11,-1.13,0.57,0.03,0.15,-0.93,0.07,-1.52,0.37,0.7,-1.22,0.16,0.38,-0.65,0.49,-0.18,-0.68,-1.52,-0.65,-0.46,-1.29\n0.33,-0.31,1.12,0.59,0.62,0.42,-1.38,0.76,-0.69,0.88,1.43,-0.14,-0.74,2.17,-1.25,-0.59,0.59,0.27,1.59,-0.47,-0.65,1.68,-0.21,-0.18,-0.1,-0.32,0.31,-0.06,-1.12,-1.64\n"
        ),
        delimiter=",",
    ).T
    design = np.genfromtxt(
        StringIO(
            "1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,1\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n1,2\n"
        ),
        delimiter=",",
    )
    results_init = np.genfromtxt(
        StringIO(
            "-0.8554888214077533,-0.2720977014782501,1.467533504177673,-0.22767956411506196,1.2785521085014837,0.14379806862053318,1.4750333827944937,-0.6570301309706671,0.2066831996074998,-0.353750513692208,0.40174695240184183,1.0849159925675056,1.8230740377826868,-0.3796132967907797,1.27432278200101,0.25734634440017745,-0.48925340880447665,0.15787257755416356,0.04938969573830959,0.3192312181063839,-0.3542994869485033,1.401448533272853,-0.3938541882847783,0.08741246846567274,-0.6510475608713524,0.22923361101684514,-0.06804635141148654,-0.8510364064773515,-1.6943910912740754,-1.7500059544823854\n0.22583899533086937,-0.23735468442585927,0.19733713734339808,-0.4421331667319863,0.03586724830483476,0.4198858825154666,-0.08304003977298911,-0.16513486823394496,-0.30042983957286806,-0.3460343445737336,-0.3558867742284205,0.3011982462662531,-0.20850076449582816,-1.0161941132088552,-0.27541891451634165,-0.38785291893731433,-0.12147153601775817,0.9332086847171666,1.6012153818454613,-1.4764027185416573,0.23315085154197657,0.4422452475355182,-0.20026735579063953,0.04360380473394784,1.2311912129895743,-0.3862511983506538,-0.7251300187432477,0.11080434946587961,1.2741080839827195,1.467848129569024\n-0.7940511553704984,0.36872650465867485,-0.696190082245086,0.7842394355421791,-0.2859846254450281,-1.2496520777731421,0.3835639668947345,0.06394998322061843,0.6354423384795286,0.43034530046034825,0.446899273641681,-0.6513965946225246,0.6871089757451906,1.8676308579518373,0.6993678988614832,-0.35853838671959254,1.3529850324988306,-0.246355558204097,1.1840747073603424,-0.40278806288570534,-0.980382841146775,1.0552239017849732,0.9738387917168745,-0.4865414099512409,-0.14227709749759898,-0.07349599449524213,0.7673735611128623,-0.27683325130457614,-1.4140680282113947,-0.2222153640576589\n1.418455323621421,1.003633642928876,-2.249906497292914,0.6558257816955055,-1.7203847145600766,-0.29827588824455953,0.043760865569417634,0.9060713852440867,0.8559072353487625,0.15882996592743862,-1.1213675611187175,-0.13675398777029954,-0.4426075110785659,-0.28963584470503034,-0.27355219556533633,-0.22900366687718224,-1.4215471940245057,0.5395008912619365,0.30620330206252744,-0.9001187449182614,-0.09048745003886999,0.7701978486901692,-1.2331890579233868,-0.0319982155355501,-0.019760633637726566,-0.37122588017711394,-1.2165536309067462,-0.9082217868619842,-0.3910853125055948,-0.8027104686077078\n0.16860548847823306,0.019631704090379926,1.069169569969778,0.4089409239787705,0.8084926808056435,0.6289431224230143,-1.2935687373997728,0.6009167939510944,-0.42509727026387234,1.0706646727858744,1.4570831601240841,-0.7309939926404594,-1.1819487005772338,1.8301112490396232,-0.6209506647651512,-0.1631607424281398,0.3494484690478879,0.3004091825038984,1.248612413250847,-0.5327275335431212,-0.7111287487922291,1.7104365790742557,0.179842226080069,-0.19838902997632976,-0.10222070429937023,-0.02570345245587259,0.10711511924305978,-0.7160399504157087,-1.5156808495142373,-0.8308129777750086\n"
        ),
        delimiter=",",
    ).T

    model = pylemur.tl.LEMUR(Y, design=design, n_embedding=3)
    model.fit(verbose=False)
    pred_init = model.predict()
    assert np.allclose(pred_init, results_init)

    ## Here is a modified version of the approach from the previous test
    # set.seed(1)
    # grouping <- sample(c(1,2), size = 30, replace = TRUE)
    # save_func(grouping)
    # fit <- align_by_grouping(fit, grouping = as.character(grouping), design = fit$design_matrix)
    # save_func(cbind(fit$alignment_coefficients[,,1], fit$alignment_coefficients[,,2]))
    # ...
    grouping = np.genfromtxt(
        StringIO("1\n2\n1\n1\n2\n1\n1\n1\n2\n2\n1\n1\n1\n1\n1\n2\n2\n2\n2\n1\n1\n1\n1\n1\n1\n1\n2\n1\n1\n2\n")
    )

    basepoint = np.genfromtxt(
        StringIO(
            "-0.38925206703492543,0.5213219572509142,-0.7231411670117558\n-0.014297729446464128,-0.3059067922758592,0.056315487696737755\n0.2455091078121989,0.48054559390052237,0.133975572527343\n0.7927765077921876,-0.20187676099522883,-0.5743821409539152\n0.39938589098238725,0.602466726782849,0.3550086203707195\n"
        ),
        delimiter=",",
    ).T
    assert np.allclose(pylemur.tl._grassmann.grassmann_angle_from_point(basepoint.T, model.base_point.T), 0)
    emb = np.genfromtxt(
        StringIO(
            "1.307607290530331,1.204126637770359,-2.334941421246343,1.202469151007156,-1.7184261449703435,-0.5734035327253721,-0.6805385978545121,1.3185253662996623,0.8741076652495349,0.8858962317444363,-0.34911335825714623,-0.9635203043565856,-1.052446567304073,1.396858005348387,-0.5172004212354832,-0.5028122904697048,-0.3322559893690832,1.050654556399191,2.1375501746096512,-1.7864259568934735,-0.38553520332697033,1.7579862599667835,-0.44888734172910766,-0.14472710389159946,0.8047342904549903,-0.4509018220855433,-0.8826656042763416,-0.7455036156569727,-0.28188072558002397,0.2106703718482004\n-1.8164798196642507,-0.477295729296259,0.4790240845360194,0.1574495019414363,0.5418966595889152,-1.218733839936566,-0.030245840455864665,-0.6590134762699248,-0.17773495609983386,0.16681479907923033,0.9577303872796369,-0.8649693035413422,0.5516152492211109,1.872319735730998,0.5176225478866773,0.1933219672043183,1.1600064890134691,-0.2055279125365953,0.8140862711968238,0.5522995484090838,-1.0056198219223997,2.0135021529189063,0.9037871448321165,-0.1835087018953936,-0.7687706326544156,0.4350873523204601,1.1121743457677276,-0.6639780098588698,-2.6747181225665626,-1.6821420702286731\n0.18948314499743618,-0.35557961492948087,1.2558867706759906,-0.06200523822352812,0.8150276978150715,0.9017861009351356,-1.7593369587152685,0.38440670001846194,-0.9259345355859308,0.7952852407963301,1.2519868131937648,-0.7881479698006301,-1.7254602945657311,1.1468879075695446,-1.1242857641811612,-0.777687817264553,1.2908968434310275,-0.7487113052869242,0.3633562246175412,-0.7434161046620548,-0.6564971373410212,-1.062751302635433,0.9113503972724165,-0.742431234492101,0.1738627261607038,-0.5327010359755776,0.46611269007910494,0.42937368327065845,0.34776880301511215,1.281474569811115\n"
        ),
        delimiter=",",
    ).T
    coef = np.genfromtxt(
        StringIO(
            "-0.39126157680942947,-0.13893603024573095,0.22217190036192688,0.2997192813183081,0.045415132726990666,-0.11388588356030183\n-1.4118852084954074,-0.12010846331372559,-0.042620766619525983,1.1685923275506187,-0.009795610498605375,0.1481261731296855\n0.4301922175223224,0.8308632945684709,-1.7460499853532059,-0.17472584078151635,-0.3588445576311046,1.1196336953167267\n0.010836267191959998,-0.020245706071854806,0.04720615470254291,-0.013801681502086402,0.009718078909566252,-0.032178386499295816\n-0.7178345111625161,-0.6102684315610786,1.1946309212817439,0.46875216448007323,0.2452095180077213,-0.7300769880872995\n"
        ),
        delimiter=",",
    )
    coef = np.stack([coef[:, 0:3], coef[:, 3:6]], axis=2)
    coef = np.einsum("ijk->jik", coef)

    wide_alignment_coef = np.genfromtxt(
        StringIO(
            "-0.09057594036322775,0.004854216024969793,0.002327313009887152,0.005758995615529929,0.05944370038362205,-0.0035991050730759615,-0.0020095175466780125,-0.005428755213860675\n-0.05277256958166519,0.042258290722341184,0.020260381752233707,0.05013483325379276,0.02699629278032626,-0.03133194479519331,-0.017493819035318634,-0.04725993134785861\n-0.07693347488863567,0.04690711963321551,0.02248922363923753,0.0556501595551681,0.04220308611767923,-0.03477876785187711,-0.019418311727825953,-0.052458990075042476\n"
        ),
        delimiter=",",
    )
    alignment_coef = np.stack([wide_alignment_coef[:, 0:4], wide_alignment_coef[:, 4:8]], axis=2)
    results = np.genfromtxt(
        StringIO(
            "-0.8554888214077531,-0.27209770147824963,1.4675335041776725,-0.22767956411506185,1.2785521085014835,0.14379806862053301,1.4750333827944933,-0.6570301309706669,0.20668319960749992,-0.3537505136922079,0.4017469524018417,1.0849159925675056,1.8230740377826868,-0.3796132967907798,1.27432278200101,0.25734634440017756,-0.4892534088044769,0.15787257755416356,0.049389695738309536,0.3192312181063839,-0.3542994869485032,1.4014485332728528,-0.3938541882847784,0.0874124684656728,-0.6510475608713524,0.22923361101684514,-0.0680463514114866,-0.8510364064773513,-1.6943910912740756,-1.7500059544823856\n0.22583899533086926,-0.2373546844258592,0.19733713734339786,-0.4421331667319863,0.03586724830483465,0.4198858825154665,-0.08304003977298917,-0.16513486823394496,-0.300429839572868,-0.34603434457373355,-0.3558867742284205,0.3011982462662531,-0.20850076449582813,-1.0161941132088552,-0.2754189145163416,-0.38785291893731433,-0.12147153601775795,0.9332086847171664,1.601215381845461,-1.476402718541657,0.23315085154197665,0.44224524753551797,-0.20026735579063937,0.043603804733947815,1.2311912129895743,-0.3862511983506537,-0.7251300187432477,0.11080434946587975,1.27410808398272,1.467848129569024\n-0.7940511553704983,0.36872650465867474,-0.6961900822450855,0.784239435542179,-0.2859846254450279,-1.249652077773142,0.3835639668947346,0.0639499832206183,0.6354423384795286,0.43034530046034813,0.446899273641681,-0.6513965946225245,0.6871089757451905,1.8676308579518373,0.6993678988614832,-0.3585383867195926,1.352985032498831,-0.24635555820409705,1.1840747073603421,-0.40278806288570546,-0.9803828411467752,1.0552239017849732,0.9738387917168745,-0.486541409951241,-0.1422770974975989,-0.07349599449524222,0.7673735611128623,-0.27683325130457614,-1.4140680282113949,-0.22221536405765868\n1.4184553236214208,1.0036336429288755,-2.2499064972929133,0.6558257816955054,-1.7203847145600764,-0.2982758882445596,0.043760865569417745,0.9060713852440865,0.8559072353487623,0.15882996592743856,-1.1213675611187173,-0.13675398777029935,-0.4426075110785658,-0.28963584470503045,-0.2735521955653362,-0.22900366687718213,-1.4215471940245057,0.5395008912619363,0.3062033020625272,-0.9001187449182614,-0.09048745003886988,0.7701978486901689,-1.2331890579233868,-0.031998215535550045,-0.019760633637726732,-0.37122588017711383,-1.2165536309067464,-0.9082217868619842,-0.3910853125055947,-0.802710468607708\n0.1686054884782331,0.019631704090379815,1.0691695699697783,0.40894092397877047,0.8084926808056437,0.6289431224230144,-1.2935687373997728,0.6009167939510944,-0.42509727026387234,1.0706646727858744,1.4570831601240841,-0.7309939926404595,-1.1819487005772338,1.8301112490396232,-0.6209506647651512,-0.16316074242813974,0.34944846904788796,0.3004091825038983,1.248612413250847,-0.5327275335431213,-0.7111287487922291,1.7104365790742553,0.179842226080069,-0.19838902997632976,-0.10222070429937029,-0.02570345245587259,0.10711511924305983,-0.7160399504157086,-1.5156808495142378,-0.8308129777750086\n"
        ),
        delimiter=",",
    ).T

    model.base_point = basepoint
    model.embedding = emb
    model.coefficients = coef
    model.align_with_grouping(grouping)
    assert np.allclose(model.alignment_coefficients, alignment_coef)

    pred_new = model.predict()
    assert np.allclose(pred_new, results)
    assert np.allclose(pred_new, pred_init)


def test_align_works():
    nelem = 200
    Y = np.random.randn(nelem, 5)
    design = np.stack([np.ones(nelem), np.random.choice(2, size=nelem)], axis=1)
    model = pylemur.tl.LEMUR(Y, design=design, n_embedding=3)
    model.fit(verbose=False)
    model_2 = model.copy()
    model_3 = model.copy()

    grouping = np.random.choice(2, nelem)
    model_2.align_with_grouping(grouping)
    model_3.align_with_harmony()
    assert np.allclose(model_2.predict(), model.predict())
    assert np.allclose(model_3.predict(), model.predict())


def _assert_lemur_model_equal(m1, m2, adata_id_equal=True):
    for k in m1.__dict__.keys():
        if k == "adata" and adata_id_equal:
            assert id(m1.adata) == id(m2.adata)
        elif k == "adata" and not adata_id_equal:
            assert id(m1.adata) != id(m2.adata)
        elif isinstance(m1.__dict__[k], pd.DataFrame):
            pd.testing.assert_frame_equal(m1.__dict__[k], m2.__dict__[k])
        elif isinstance(m1.__dict__[k], np.ndarray):
            assert np.array_equal(m1.__dict__[k], m2.__dict__[k])
        else:
            assert m1.__dict__[k] == m2.__dict__[k]
