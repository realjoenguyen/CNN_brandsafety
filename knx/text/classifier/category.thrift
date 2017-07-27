namespace java category
namespace py category

service Category {
        void ping(),
        map<string,string> getCategory(1:string data),
        list<map<string,string>> getMultiCategory(1:string data)
}
