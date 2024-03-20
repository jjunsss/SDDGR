import sys

def get_models(model_name, args, num_classes, current_class):
    # build model
    if model_name == 'deform_detr':
        from .deform_detr import build_model
    elif model_name == 'dn_detr':
        from .dn_detr import build_model
    # elif model_name == ...:
    #     모델 계속 추가

    return build_model(args, num_classes, current_class)

def inference_model(args, model, samples, targets=None, teacher_attn=None, eval=False):
    model.to(args.device)
    
    if args.model_name == 'deform_detr':
        return model(samples, teacher_attn)
    
    elif args.model_name == 'dn_detr':
        assert targets is not None
        if eval:
            dn_args = 0
        else:
            dn_args=(targets, args.scalar, args.label_noise_scale, args.box_noise_scale, args.num_patterns)
            if args.contrastive is not False:
                dn_args += (args.contrastive,)
        return model(samples, dn_args)
    
    sys.exit(0)