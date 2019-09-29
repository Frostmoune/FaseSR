import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srcos':
        from .SRCos_model import SRCosModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'srcosragan':
        from .SRCosRaGAN_model import SRCosRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'srcos_discriminator':
        from .SRCosDiscriminator_model import SRCosDiscriminatorModel as M
    elif model == 'srracos_discriminator':
        from .SRRaCosDiscriminator_model import SRRaCosDiscriminatorModel as M
    elif model == 'srcos_style':
        from .SRCosStyle_model import SRCosStyleModel as M
    elif model == 'sr_style':
        from .SRStyle_model import SRStyleModel as M
    elif model == 'srcos_stylegan':
        from .SRCosStyleRaGAN_model import SRCosStyleRaGANModel as M
    elif model == 'sr2hr':
        from .SR2HR_model import SR2HRModel as M
    elif model == 'sr2hrgan':
        from .SR2HRGAN_model import SR2HRGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
