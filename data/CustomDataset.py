def get_dataset(opt):
    dataset = None
    if opt.dataset_mode == 'visDrone':
        from data.visDroneDataset import VisDrone
        dataset = VisDrone(opt)
    else:
        raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)

    # if opt.dataset_mode == 'list':
    #     from data.list_dataset import ListDataset
    #     dataset = ListDataset(opt.dataroot, img_size=opt.img_size, augment=(not opt.noagument), multiscale=(not opt.nomultiscale))
    # elif opt.dataset_mode == 'crop':
    #     from data.crop_dataset import CropDataset
    #     dataset = CropDataset(opt.dataroot, img_size=opt.img_size, augment=(not opt.noagument), multiscale=(not opt.nomultiscale))
    # elif opt.dataset_mode == 'test':
    #     from data.image_folder import ImageFolder
    #     dataset = ImageFolder(opt.dataroot, img_size=opt.img_size)
    # elif opt.dataset_mode == 'test_crop':
    #     from data.image_folder import TestCrop
    #     dataset = TestCrop(opt.dataroot, img_size=opt.img_size)
    # if opt.dataset_mode == 'stage_one_train':
    #     from data.two_stage_dataset import StageOneTrain
    #     dataset = StageOneTrain(opt.dataroot, img_size=opt.img_size)
    # else:
    #     raise NotImplementedError('the dataset [%s] is not implemented' % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset