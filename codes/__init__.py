import model_predict as mp
import sys
import warnings
warnings.filterwarnings(action='ignore')
mmp = mp.ModelPredict()

if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
        try:
            result_df = mmp.predict_check_result(filename=arg1)
        except:
            arg1 = int(arg1)
            try:
                arg2 = int(sys.argv[2])
                result_df = mmp.predict_check_result(start=arg1, end=arg2)
            except:
                result_df = mmp.predict_check_result(end=arg1)

        print(result_df)

    except:
        try:
            result_df = mmp.predict_check_result()
            print(result_df.head())
            print(f"Total number of data: {len(result_df)}")
        except:
            print("Please check the type of input data."
                  "It should be a filename(str) or start, end row number(int).")