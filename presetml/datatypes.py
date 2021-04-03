import attr


@attr.s(frozen=True)
class GANDiscriminatorParameters:
    learning_rate = attr.ib(
        default=0.00005, type=float, validator=attr.validators.instance_of(float)
    )
    hidden_layers = attr.ib(
        default=4, type=int, validator=attr.validators.instance_of(int)
    )
    input_noise_stdev = attr.ib(
        default=0.01, type=float, validator=attr.validators.instance_of(float)
    )
    category_input_layers = attr.ib(
        default=2, type=int, validator=attr.validators.instance_of(int)
    )
    layer_size = attr.ib(
        default=400, type=int, validator=attr.validators.instance_of(int)
    )
    category_layer_size = attr.ib(
        default=400, type=int, validator=attr.validators.instance_of(int)
    )
    dropout = attr.ib(
        default=0.4, type=float, validator=attr.validators.instance_of(float)
    )
    relu_alpha = attr.ib(
        default=0.2, type=float, validator=attr.validators.instance_of(float)
    )


@attr.s(frozen=True)
class GANGeneratorParameters:
    learning_rate = attr.ib(
        default=0.00005, type=float, validator=attr.validators.instance_of(float)
    )
    latent_dim = attr.ib(
        default=100, type=int, validator=attr.validators.instance_of(int)
    )
    hidden_layers = attr.ib(
        default=1, type=int, validator=attr.validators.instance_of(int)
    )
    category_input_layers = attr.ib(
        default=2, type=int, validator=attr.validators.instance_of(int)
    )
    layer_size = attr.ib(
        default=400, type=int, validator=attr.validators.instance_of(int)
    )
    category_layer_size = attr.ib(
        default=400, type=int, validator=attr.validators.instance_of(int)
    )
    dropout = attr.ib(
        default=0.4, type=float, validator=attr.validators.instance_of(float)
    )
    relu_alpha = attr.ib(
        default=0.2, type=float, validator=attr.validators.instance_of(float)
    )


@attr.s(frozen=True)
class GANConfig:
    num_epochs = attr.ib(
        default=100, type=int, validator=attr.validators.instance_of(int)
    )
    batch_size = attr.ib(
        default=256, type=int, validator=attr.validators.instance_of(int)
    )
    discriminator_update_multiplier = attr.ib(
        default=5, type=int, validator=attr.validators.instance_of(int)
    )
    cgan_labels = attr.ib(default=None)

    discriminator_parameters = attr.ib(
        default=GANDiscriminatorParameters(),
        type=GANDiscriminatorParameters,
        validator=attr.validators.instance_of(GANDiscriminatorParameters),
    )
    generator_parameters = attr.ib(
        default=GANGeneratorParameters(),
        type=GANGeneratorParameters,
        validator=attr.validators.instance_of(GANGeneratorParameters),
    )
