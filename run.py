import os
from imageatm.components import DataPrep
from imageatm.components import Training
from imageatm.components import Evaluation
import keras.backend as K
K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    dp = DataPrep(
        image_dir='./cats_and_dogs/train',
        samples_file='data.json',
        job_dir='./sample'
    )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    dp.run(resize=True)
    # 모델을 학습시키는 부분
    trainer = Training(dp.image_dir, dp.job_dir,
                       epochs_train_dense=3, epochs_train_all=1)
    trainer.run()
    # 학습모델의 평가 부분
    e = Evaluation(image_dir=dp.image_dir, job_dir=dp.job_dir)
    e.run()
